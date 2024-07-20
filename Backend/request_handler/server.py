import uuid
import threading
from flask import Flask, request, jsonify
import os
import base64
import re
from flask_cors import CORS
from utils.contour_filtering import contour_filter
from learner.resnet18_vgg16 import HandwrittenSymbolsClassifier
import torchvision.transforms as transforms
from PIL import Image
import glob
import cv2
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IS_MODEL_TRAINING = False

model_name = "resnet34"  # or 'resnet'

classifier = HandwrittenSymbolsClassifier(
    root_dir="./learner/datasets/extracted_images/",
    epochs=5,
    batch_size=32,  # for vgg use 32 otherwise 64
    model_type=f"{model_name}",

)
 

def load_db():
    print("Current working directory:", os.getcwd())
    try:
        with open('db/data.json', 'r') as f:
            data = json.load(f)
            # print("Data loaded:", data)
            return data
    except Exception as e:
        print("Failed to load data:", e)

    

def load_or_train_model():
    global IS_MODEL_TRAINING
    with train_lock:
        if not IS_MODEL_TRAINING:
            IS_MODEL_TRAINING = True
            threading.Thread(target=train_model).start()

def train_model():
    global IS_MODEL_TRAINING
    try:
        classifier.load_model(f"./learner/models/prod_{model_name}.torch")
        print("Model loaded successfully")
    except Exception as e:
        try:
            classifier.train()
            classifier.save_model("./learner/models/", f"prod_{model_name}.torch")
            classifier.load_model(f"./learner/models/prod_{model_name}.torch")
            print("Model trained and saved successfully")
        except Exception as e:
            print(e)
    finally:
        IS_MODEL_TRAINING = False

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def get_home():
    return "Server is running"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

lock = threading.Lock()
train_lock = threading.Lock()
prediction_data = {}
incomplete_predictions = set()

def run_prediction(file_id, filename):
    global IS_MODEL_TRAINING
    with train_lock:
        if IS_MODEL_TRAINING:
            with lock:
                prediction_data[file_id] = "Error: Model is still training"
                incomplete_predictions.remove(file_id)
            return

    try:
        prediction_text,prediction_latex = schedule_request(filename, file_id)
        with lock:
            prediction_data[file_id] = [prediction_text,prediction_latex]
            incomplete_predictions.remove(file_id)
    except Exception as e:
        with lock:
            prediction_data[file_id] = f"Error: {str(e)}"
            incomplete_predictions.remove(file_id)

def schedule_request(filename, file_id):
    prediction_text = ""
    prediction_latex = ""
    db = load_db()
    contour_filter(file_id)
    image_paths = glob.glob(f"extracted_characters/{file_id}/**/*", recursive=True)
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(image, (45, 45), interpolation=cv2.INTER_AREA)
        if len(resized_image.shape) == 3:
            gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = resized_image

        _, bw_image = cv2.threshold(gray_image, 220, 256, cv2.THRESH_BINARY)

        cv2.imwrite(image_path, bw_image)
        prediction = classifier.predict(image_path=image_path)
        prediction_text += prediction
        # prediction_latex += {db[prediction]}
        prediction_latex+= db.get(prediction, "")
        print("Prediction: ", prediction_text)
        print("Prediction Latex: ", prediction_latex)
    return prediction_text,prediction_latex

@app.route('/postCanvas', methods=['POST'])
def upload_canvas():
    try:
        if 'imageData' not in request.json:
            return jsonify({'error': 'No imageData part in the request'}), 400

        data_url = request.json['imageData']
        img_str = re.search(r'base64,(.*)', data_url).group(1)
        img_data = base64.b64decode(img_str)
        file_id = uuid.uuid4().hex
        filename = f"{file_id}.png"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        with open(file_path, 'wb') as f:
            f.write(img_data)
        
        with lock:
            incomplete_predictions.add(file_id)
        
        threading.Thread(target=run_prediction, args=(file_id, filename)).start()
        return jsonify({'message': 'Processing', 'filename': filename, 'file_id': file_id}), 201
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/prediction_status/<string:file_id>', methods=['GET'])
def get_prediction_status(file_id):
    with lock:
        if file_id in prediction_data:
            return jsonify({'file_id': file_id, 'status': 'completed', 'result': {'plain_text':prediction_data[file_id][0], 'latex_text':prediction_data[file_id][1]}})
        elif file_id in incomplete_predictions:
            return jsonify({'file_id': file_id, 'status': 'incomplete', 'message': 'Prediction is still in progress'}), 200
        else:
            return jsonify({'status': -1, 'error': 'Prediction data not found for the file ID'}), 404

@app.route("/model_details", methods=['GET'])
def model_details():
    global model_name
    try:
        with open(f'learner/models/prod_{model_name}.torch.metadata.json', 'r') as f:
            data = json.load(f)
            return jsonify(data)
    except Exception as e:
        print("Failed to load data:", e)
        return jsonify({"error": e})

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
load_or_train_model()

tokenizer = None
llm = None


def download_and_configure_llm():

    llm_name = "deepseek-ai/deepseek-math-7b-base"
    llm_directory = "./models/llm"
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    llm = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.bfloat16, device_map="auto")
    llm.generation_config = GenerationConfig.from_pretrained(model_name)
    llm.generation_config.pad_token_id = llm.generation_config.eos_token_id
    if not os.path.exists(llm_directory):
        os.makedirs(llm_directory)
    
    tokenizer_path = os.path.join(llm_directory, "tokenizer")
    model_path = os.path.join(llm_directory, "model")
    
    if not os.path.exists(tokenizer_path) or not os.path.exists(model_path):
        print("Downloading and configuring LLM...")
        tokenizer = AutoTokenizer.from_pretrained(llm_name)
        llm = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.bfloat16, device_map="auto")
        llm.generation_config = GenerationConfig.from_pretrained(llm_name)
        llm.generation_config.pad_token_id = llm.generation_config.eos_token_id
        
        tokenizer.save_pretrained(tokenizer_path)
        llm.save_pretrained(model_path)
        print("LLM downloaded and configured successfully.")
    else:
        print("LLM already downloaded and configured.")


@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    text = data.get("text", "")
    inputs = tokenizer(text, return_tensors="pt")
    outputs = llm.generate(**inputs.to(llm.device), max_new_tokens=100)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"result": result})


@app.route('/download_llm', methods=['GET'])
def download_llm():
    try:
        threading.Thread(target=download_and_configure_llm).start()
        return jsonify({'status': 'success', 'message': 'LLM download and configuration initiated'}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500