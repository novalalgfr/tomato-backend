import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from utils.model_loader import load_model
from utils.image_processing import process_image

UPLOAD_FOLDER    = 'uploads'
STATIC_FOLDER    = 'static'
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

MESSAGE_MAP = {
    'no_detection'    : 'Tidak ada penyakit daun tomat yang terdeteksi',
    'disease_detected': 'Analysis Complete',
}

@app.route('/')
def home():
    return jsonify({
        "status" : "running",
        "message": "Tomato.Logy AI Backend is Operational."
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"status": "error", "error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"status": "error", "error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"status": "error", "error": "File type not allowed. Use JPG/PNG."}), 400

    if model is None:
        return jsonify({"status": "error", "error": "Model AI belum siap."}), 500

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result_data     = process_image(model, filepath, static_folder=STATIC_FOLDER)
        msg_flag        = result_data.get('message', 'disease_detected')
        user_message    = MESSAGE_MAP.get(msg_flag, 'Analysis Complete')
        base_url        = request.host_url.rstrip('/')
        detect_url_raw  = result_data.get('detect_url')
        detect_url_full = f"{base_url}{detect_url_raw}" if detect_url_raw else None

        return jsonify({
            "status"          : "success",
            "message"         : user_message,
            "result_type"     : msg_flag,
            "detect_url"      : detect_url_full,
            "detections_count": result_data['detections'],
            "details"         : result_data['details'],
        })

    except Exception as e:
        print(f"Error Processing: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4010, debug=False)