import os
import cv2
import uuid

def process_image(model, image_path, static_folder='static'):
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError("Gambar tidak valid/corrupt")
    
    results = model.predict(original_img, conf=0.25)
    result = results[0]
    boxes = result.boxes
    
    annotated_img = result.plot()
    
    unique_id = str(uuid.uuid4())[:8]
    
    results_dir = os.path.join(static_folder, 'results')
    os.makedirs(results_dir, exist_ok=True)

    detect_filename = f"detect_{unique_id}.jpg"
    detect_path = os.path.join(results_dir, detect_filename)

    cv2.imwrite(detect_path, annotated_img)

    base_url = "/static/results"
    
    return {
        'detect_url': f"{base_url}/{detect_filename}",
        'detections': len(boxes)
    }