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
        prediction_text = schedule_request(filename, file_id)
        with lock:
            prediction_data[file_id] = prediction_text
            incomplete_predictions.remove(file_id)
    except Exception as e:
        with lock:
            prediction_data[file_id] = f"Error: {str(e)}"
            incomplete_predictions.remove(file_id)

def schedule_request(filename, file_id):
    prediction_text = ""
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
    return prediction_text

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
            return jsonify({'file_id': file_id, 'status': 'completed', 'result': prediction_data[file_id]})
        elif file_id in incomplete_predictions:
            return jsonify({'file_id': file_id, 'status': 'incomplete', 'message': 'Prediction is still in progress'}), 200
        else:
            return jsonify({'status': -1, 'error': 'Prediction data not found for the file ID'}), 404

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
load_or_train_model()