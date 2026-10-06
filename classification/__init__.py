"""
Classification package for Sheep Health and Disease Detection.
"""
from .preprocessing import preprocess_image, preprocess_image_tta, IMG_SIZE
from .preprocessor import preprocess_image as prep_img
from .class_names import (
    CLASS_NAMES,
    UNCERTAINTY_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    DISPLAY_NAMES,
    SHORT_NAMES,
    TELUGU_NAMES,
    AFFECTED_REGIONS
)
from .metadata import (
    build_prediction_context
)
from .predictor import load_model, predict, predict_and_analyze, MODEL_PATH, BASE_DIR


predict_sheep_image = predict

__all__ = [
    "preprocess_image",
    "preprocess_image_tta",
    "IMG_SIZE",
    "CLASS_NAMES",
    "UNCERTAINTY_THRESHOLD",
    "LOW_CONFIDENCE_THRESHOLD",
    "DISPLAY_NAMES",
    "SHORT_NAMES",
    "TELUGU_NAMES",
    "AFFECTED_REGIONS",
    "build_prediction_context",
    "load_model",
    "predict",
    "predict_sheep_image",
    "predict_and_analyze",
    "MODEL_PATH",
    "BASE_DIR"
]

