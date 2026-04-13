import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

POSITIVE_LABELS = [
    "a tomato leaf",
    "a diseased tomato leaf",
    "a healthy tomato plant leaf",
    "tomato foliage",
]

NEGATIVE_LABELS = [
    "a mango leaf",
    "mango tree leaves",
    "a long oval shiny leaf",
    "a non-tomato plant leaf",
    "a random plant leaf",
    "a tree leaf",
    "a cassava leaf",
    "food",
    "an animal",
    "a human",
    "a document",
    "a building",
]

ALL_LABELS = POSITIVE_LABELS + NEGATIVE_LABELS

CONFIDENCE_THRESHOLD = 0.40

_clip_model     = None
_clip_processor = None


def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        print("[INFO] Loading CLIP model untuk validasi gambar...")
        _clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()
        print("[INFO] CLIP model loaded.")
    return _clip_model, _clip_processor


def _validate_pil(image: Image.Image) -> dict:
    model, processor = _load_clip()

    inputs = processor(
        text=ALL_LABELS,
        images=image,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        outputs          = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs            = logits_per_image.softmax(dim=1)[0]

    n_pos          = len(POSITIVE_LABELS)
    positive_score = probs[:n_pos].sum().item()
    negative_score = probs[n_pos:].sum().item()

    best_idx   = probs.argmax().item()
    best_label = ALL_LABELS[best_idx]
    best_conf  = probs[best_idx].item()

    is_valid = (positive_score >= CONFIDENCE_THRESHOLD) and (positive_score > negative_score)

    return {
        "valid"          : is_valid,
        "label"          : best_label,
        "confidence"     : round(best_conf, 4),
        "positive_score" : round(positive_score, 4),
        "reason"         : (
            f"Dikenali sebagai daun tomat (skor: {positive_score:.2f})."
            if is_valid else
            f"Bukan daun tomat — label terdekat: '{best_label}' (skor: {best_conf:.2f})."
        ),
    }


def is_tomato_leaf_crop(image: Image.Image) -> bool:
    result = _validate_pil(image)
    print(
        f"[VALIDATOR] crop → valid={result['valid']} | "
        f"label='{result['label']}' | pos_score={result['positive_score']}"
    )
    return result["valid"]


def is_tomato_leaf(image_path: str) -> dict:
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {
            "valid"      : False,
            "label"      : "unknown",
            "confidence" : 0.0,
            "reason"     : f"Gambar tidak bisa dibuka: {e}",
        }
    return _validate_pil(image)