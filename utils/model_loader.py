import os
from ultralytics import YOLO

_model = None

def load_model(model_path='models/best.pt'):
    global _model
    
    if _model is not None:
        return _model

    if not os.path.exists(model_path):
        abs_path = os.path.abspath(model_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"ERROR: Model file tidak ditemukan di: {model_path} atau {abs_path}")

    print(f" [INFO] Loading YOLOv11 Model from: {model_path}...")
    
    try:
        _model = YOLO(model_path)
        print(" [INFO] Model loaded successfully!")
    except Exception as e:
        print(f" [ERROR] Gagal memuat model: {e}")
        raise e

    return _model