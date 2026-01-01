import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from utils.model_loader import load_model
from utils.image_processing import process_image

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__, static_folder=STATIC_FOLDER)
CORS(app) 

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

try:
    model = load_model('models/best.pt') 
except Exception as e:
    print(f"WARNING: Model gagal diload. Error: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return jsonify({"status": "running", "message": "Tomato.Logy AI Backend is Operational."})

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"status": "error", "error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            if model is None:
                return jsonify({"status": "error", "error": "Model AI belum siap."}), 500

            result_data = process_image(model, filepath, static_folder=STATIC_FOLDER)

            base_url = "http://127.0.0.1:5000"
            
            return jsonify({
                "status": "success",
                "message": "Analysis Complete",
                "detect_url": f"{base_url}{result_data['detect_url']}",
                "detections_count": result_data['detections']
            })

        except Exception as e:
            print(f"Error Processing: {e}")
            return jsonify({"status": "error", "error": str(e)}), 500
        
    return jsonify({"status": "error", "error": "File type not allowed. Use JPG/PNG."}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)