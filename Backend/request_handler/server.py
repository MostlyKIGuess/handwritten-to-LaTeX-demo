import uuid
from flask import Flask, request, jsonify
import os
import base64
import re
from flask_cors import CORS
from utils.contour_filtering import contour_filter

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def get_home():
    return "Server is running"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
        contour_filter(file_id)
        return jsonify({'message': 'Image uploaded successfully', 'filename': filename}), 201
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
