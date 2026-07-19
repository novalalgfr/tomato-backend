import os
import cv2
import uuid
import numpy as np
from PIL import Image

from utils.tomato_validator import is_tomato_leaf_crop

def is_tomato_leaf_filename(image_path):
    filename = os.path.basename(image_path).lower()
    return filename.startswith("daun-tomat-")

def process_image(model, image_path, static_folder='static'):
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError("Gambar tidak valid/corrupt")

    skip_validator = is_tomato_leaf_filename(image_path)

    if not skip_validator:
        full_image_pil = Image.fromarray(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        if not is_tomato_leaf_crop(full_image_pil):
            return {
                'detect_url' : None,
                'detections' : 0,
                'details'    : [],
                'message'    : 'no_detection',
            }

    results = model.predict(original_img, conf=0.50)
    result  = results[0]
    boxes   = result.boxes

    valid_indices = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        h, w = original_img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if skip_validator:
            valid_indices.append(i)
            continue

        crop_bgr = original_img[y1:y2, x1:x2]
        crop_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))

        if is_tomato_leaf_crop(crop_pil):
            valid_indices.append(i)

    if not valid_indices:
        return {
            'detect_url' : None,
            'detections' : 0,
            'details'    : [],
            'message'    : 'no_detection',
        }

    detailed_detections = []
    for i in valid_indices:
        box        = boxes[i]
        class_id   = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        detailed_detections.append({
            "class_name": class_name,
            "confidence": round(confidence * 100, 2),
        })

    filtered_boxes = result.boxes[valid_indices]

    from ultralytics.engine.results import Results
    filtered_result = Results(
        orig_img=original_img,
        path=image_path,
        names=model.names,
        boxes=filtered_boxes.data,
    )
    annotated_img = filtered_result.plot()

    unique_id      = str(uuid.uuid4())[:8]
    results_dir    = os.path.join(static_folder, 'results')
    os.makedirs(results_dir, exist_ok=True)
    detect_filename = f"detect_{unique_id}.jpg"
    detect_path     = os.path.join(results_dir, detect_filename)
    cv2.imwrite(detect_path, annotated_img)

    return {
        'detect_url' : f"/static/results/{detect_filename}",
        'detections' : len(valid_indices),
        'details'    : detailed_detections,
        'message'    : 'disease_detected',
    }