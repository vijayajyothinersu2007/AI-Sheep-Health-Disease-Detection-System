"""
Model loading and prediction inference engine.
"""
from pathlib import Path
from typing import List, Tuple, Optional, Any, Dict
import numpy as np
from PIL import Image

from .preprocessing import preprocess_image, preprocess_image_tta
from .metadata import CLASS_NAMES, build_prediction_context

# Resolve model path relative to project root
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / 'sheep_disease_mobilenetv2.keras'

_CACHED_MODEL = None

def load_model(model_path: Optional[Path] = None):
    """
    Loads the trained MobileNetV2 sheep disease model safely.
    Caches the loaded model in memory for fast subsequent inferences.
    """
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL
        
    path = model_path or MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}")
        
    try:
        import keras
        _CACHED_MODEL = keras.models.load_model(str(path))
    except Exception as e_keras:
        h5_path = path.with_suffix('.h5')
        if h5_path.exists() and str(h5_path) != str(path):
            try:
                import keras
                _CACHED_MODEL = keras.models.load_model(str(h5_path))
            except Exception:
                pass
        if _CACHED_MODEL is None:
            try:
                import tensorflow as tf
                _CACHED_MODEL = tf.keras.models.load_model(str(path))
            except Exception:
                if h5_path.exists():
                    import tensorflow as tf
                    _CACHED_MODEL = tf.keras.models.load_model(str(h5_path))
                else:
                    raise e_keras
        
    return _CACHED_MODEL

def predict(image: Image.Image, model: Optional[Any] = None, *args, **kwargs) -> List[Tuple[str, float]]:
    """
    Runs inference on an uploaded PIL Image with multi-view test-time augmentation (TTA).
    Evaluates canonical view, aspect-ratio preserving central crop, and horizontal mirrors
    to overcome lesion distortion, orientation bias, and camera angle variations.
    Returns ranked predictions as a list of (class_name, softmax_probability) tuples sorted descending.
    """
    use_tta = kwargs.get('use_tta', True) if 'use_tta' in kwargs else (args[0] if len(args) > 0 else True)
    if model is None:
        model = load_model()
        
    if use_tta:
        batch = preprocess_image_tta(image)
        raw_batch_preds = model.predict(batch, verbose=0)
        
        # Weighted multi-view ensemble:
        # 35% Canonical full view + 35% Aspect-preserving center crop + 15% H-flip full + 15% H-flip crop
        weights = np.array([0.35, 0.35, 0.15, 0.15], dtype=np.float32)
        raw_preds = np.average(raw_batch_preds, axis=0, weights=weights)
    else:
        img_arr = preprocess_image(image)
        raw_preds = model.predict(img_arr, verbose=0)[0]
    
    # Ensure calibrated softmax probabilities
    if not np.isclose(np.sum(raw_preds), 1.0, atol=1e-2) or np.any(raw_preds < 0):
        exp_preds = np.exp(raw_preds - np.max(raw_preds))
        probabilities = exp_preds / np.sum(exp_preds)
    else:
        probabilities = raw_preds / np.sum(raw_preds)
        
    ranked = sorted(
        [(CLASS_NAMES[i], float(probabilities[i])) for i in range(len(CLASS_NAMES))],
        key=lambda x: x[1],
        reverse=True
    )
    return ranked

def predict_and_analyze(image: Image.Image, model: Optional[Any] = None) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
    """
    Performs full prediction and generates structured prediction context.
    """
    ranked = predict(image, model)
    context = build_prediction_context(ranked)
    return ranked, context
