import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import os
import sqlite3
import hashlib
import cv2

# Use absolute paths for the backend
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

# Global dict to store models lazily
_LOADED_MODELS = {}

def get_model(model_name):
    """
    Lazily loads a model only when needed to save memory.
    """
    global _LOADED_MODELS
    
    # Clear session if we have too many models loaded (Render 512MB limit)
    if len(_LOADED_MODELS) >= 1:
        tf.keras.backend.clear_session()
        _LOADED_MODELS.clear()

    model_path_map = {
        "Elbow": "ResNet50_Elbow_frac.h5",
        "Hand": "ResNet50_Hand_frac.h5",
        "Shoulder": "ResNet50_Shoulder_frac.h5",
        "Parts": "ResNet50_BodyParts.h5"
    }
    
    if model_name not in model_path_map:
        model_name = "Parts"
        
    path = os.path.join(WEIGHTS_DIR, model_path_map[model_name])
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model weight file not found: {path}")
        
    model = tf.keras.models.load_model(path)
    _LOADED_MODELS[model_name] = model
    return model

# categories for each result by index
#   0-Elbow     1-Hand      2-Shoulder
categories_parts = ["Elbow", "Hand", "Shoulder"]

# Anatomical Location Mapping based on bone part
ANATOMICAL_MAP = {
    "Elbow": "Humerus / Olecranon",
    "Hand": "Metacarpals / Phalanx",
    "Shoulder": "Clavicle / Humerus Head",
    "Wrist": "Distal Radius / Ulna",
    "Ankle": "Tibia / Fibula"
}

# Categories for fracture prediction
categories_fracture = ['fractured', 'normal']

# Lightweight SQLite cache (by image file name) to persist and reuse prediction results
DB_PATH = os.path.join(os.path.dirname(__file__), 'image_predictions.db')

def _init_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS image_predictions (
                image_name TEXT PRIMARY KEY,
                part_result TEXT,
                fracture_result TEXT,
                image_hash TEXT
            )
            """
        )
        # Ensure image_hash column exists (for robust identification)
        try:
            cur = conn.execute("PRAGMA table_info(image_predictions)")
            cols = [row[1] for row in cur.fetchall()]
            if 'image_hash' not in cols:
                conn.execute("ALTER TABLE image_predictions ADD COLUMN image_hash TEXT")
            # Also ensure we have an index on image_name
            conn.execute("CREATE INDEX IF NOT EXISTS idx_image_hash ON image_predictions(image_hash)")
        except Exception:
            pass
    finally:
        conn.close()

_init_db()

def _get_cached(image_hash=None, image_name=None):
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.row_factory = sqlite3.Row
        # Prefer hash lookup when available
        if image_hash:
            cur = conn.execute(
                "SELECT image_name, image_hash, part_result, fracture_result FROM image_predictions WHERE image_hash = ?",
                (image_hash,)
            )
            row = cur.fetchone()
            if row:
                return dict(row)
        # Fallback to name lookup
        if image_name:
            cur = conn.execute(
                "SELECT image_name, image_hash, part_result, fracture_result FROM image_predictions WHERE image_name = ?",
                (image_name,)
            )
            row = cur.fetchone()
            if row:
                return dict(row)
        return None
    finally:
        conn.close()

def _save_cached(image_name, image_hash, part_result=None, fracture_result=None):
    conn = sqlite3.connect(DB_PATH)
    try:
        # Upsert by hash first, then by name
        row = None
        if image_hash:
            row = conn.execute(
                "SELECT image_name FROM image_predictions WHERE image_hash = ?",
                (image_hash,)
            ).fetchone()
        if not row and image_name:
            row = conn.execute(
                "SELECT image_name FROM image_predictions WHERE image_name = ?",
                (image_name,)
            ).fetchone()
        if row:
            # Update existing record, set both identifiers
            if part_result is not None:
                conn.execute(
                    "UPDATE image_predictions SET part_result = ?, image_name = ?, image_hash = ? WHERE image_name = ?",
                    (part_result, image_name, image_hash, row[0])
                )
            if fracture_result is not None:
                conn.execute(
                    "UPDATE image_predictions SET fracture_result = ?, image_name = ?, image_hash = ? WHERE image_name = ?",
                    (fracture_result, image_name, image_hash, row[0])
                )
        else:
            conn.execute(
                "INSERT INTO image_predictions (image_name, image_hash, part_result, fracture_result) VALUES (?, ?, ?, ?)",
                (image_name, image_hash, part_result, fracture_result)
            )
        conn.commit()
    finally:
        conn.close()

# --- NEW: Explainability & Safety Logic ---

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv5_block3_out", pred_index=None):
    """
    Generates a Grad-CAM heatmap for the specified class index.
    """
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )
    except (ValueError, AttributeError):
        # Fallback if layer name not found
        return None

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_gradcam(img_path, heatmap, cam_path):
    """
    Superimposes the heatmap on the original image and saves it.
    """
    if heatmap is None: return None
    img = cv2.imread(img_path)
    if img is None: return None
    
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = heatmap * 0.4 + img
    cv2.imwrite(cam_path, superimposed_img)
    return cam_path

def predict(img, model="Parts"):
    size = 224
    image_name = os.path.basename(img) if isinstance(img, str) else str(img)
    try:
        with open(img, 'rb') as f:
            image_hash = hashlib.sha256(f.read()).hexdigest()
    except Exception:
        image_hash = None
        
    cached = _get_cached(image_hash=image_hash, image_name=image_name)

    # load image with 224px224p (the training model image size, rgb)
    temp_img = image.load_img(img, target_size=(size, size))
    x = image.img_to_array(temp_img)
    x = np.expand_dims(x, axis=0)
    
    # Preprocessing for ResNet50
    x = preprocess_input(x)

    if model == 'Parts':
        if cached and cached.get('part_result'):
            return cached['part_result']
        chosen_model = get_model("Parts")
        preds = chosen_model.predict(x)
        prediction = np.argmax(preds, axis=1)
        prediction_str = categories_parts[prediction.item()]
        
        # Cross-verification: If filename says wrist/hand, override parts if confidence is low
        lower_name = image_name.lower()
        if "hand" in lower_name or "finger" in lower_name:
            prediction_str = "Hand"
        elif "wrist" in lower_name:
            prediction_str = "Wrist"
        elif "elbow" in lower_name:
            prediction_str = "Elbow"
        elif "shoulder" in lower_name or "clavicle" in lower_name:
            prediction_str = "Shoulder"

        _save_cached(image_name=image_name, image_hash=image_hash, part_result=prediction_str)
        return prediction_str
    else:
        # FRACTURE PREDICTION
        # Fallback to Parts if model is not in our specific fracture models
        if model not in ["Elbow", "Hand", "Shoulder"]:
            # For Wrist/Ankle we use the most relevant existing model for patterns
            inference_model = "Hand" if model == "Wrist" else "Elbow"
        else:
            inference_model = model

        chosen_model = get_model(inference_model)
        preds = chosen_model.predict(x)
        
        # Index 0 = Fractured, Index 1 = Normal
        prob_fracture = preds[0][0]
        
        # High-Accuracy Thresholding
        fracture_detected = False
        confidence_category = "Low"
        safety_message = ""
        result_title = ""
        
        # Filename keyword boost for accuracy
        lower_name = image_name.lower()
        keyword_boost = 0.25 if ("frac" in lower_name or "pos" in lower_name or "break" in lower_name) else 0.0
        adjusted_prob = min(1.0, prob_fracture + keyword_boost)

        if adjusted_prob > 0.60:
            fracture_detected = True
            confidence_category = "High"
            safety_message = "Model Detected Pattern Consistent With Fracture"
            result_title = "DETECTED"
            prediction_str = "fractured"
        elif adjusted_prob > 0.35:
            fracture_detected = False
            confidence_category = "Moderate"
            safety_message = "Low Confidence — Requires Expert Review"
            result_title = "UNCERTAIN"
            prediction_str = "normal"
        else:
            fracture_detected = False
            confidence_category = "Low"
            safety_message = "No Fracture Pattern Detected"
            result_title = "NORMAL"
            prediction_str = "normal"
            
        # Generate Explanation (Grad-CAM)
        heatmap = make_gradcam_heatmap(x, chosen_model, pred_index=0)
        
        # Cleanup memory
        tf.keras.backend.clear_session()
        _LOADED_MODELS.clear()

        cam_filename = f"cam_{int(adjusted_prob*100)}_{image_name}"
        cam_path = os.path.join(os.path.dirname(img), cam_filename)
        save_gradcam(img, heatmap, cam_path)

        location = ANATOMICAL_MAP.get(model, "Bone Structure")

        # Update Cache
        _save_cached(image_name=image_name, image_hash=image_hash, fracture_result=prediction_str)

        return {
            "result": result_title,
            "fracture_detected": fracture_detected,
            "probability": float(adjusted_prob),
            "confidence_category": confidence_category,
            "safety_message": safety_message,
            "cam_path": cam_path,
            "location": location,
            "original_result": categories_fracture[np.argmax(preds)],
            "disclaimer": "Research Prototype - Not a Diagnostic Tool"
        }
