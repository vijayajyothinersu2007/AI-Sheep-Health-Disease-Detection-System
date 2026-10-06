"""
Image preprocessing module for Sheep Disease Detection.
Preserves exact original model preprocessing specifications (224x224 RGB, float32).
"""
import numpy as np
from PIL import Image

IMG_SIZE = (224, 224)

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Prepares a single PIL Image for MobileNetV2 inference.
    Input image is converted to RGB, resized to (224, 224), and expanded into shape (1, 224, 224, 3) float32 in [0, 255].
    """
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL Image, got {type(image)}")
    
    img_resized = image.convert('RGB').resize(IMG_SIZE, Image.Resampling.BILINEAR)
    img_arr = np.expand_dims(np.asarray(img_resized, dtype=np.float32), axis=0)
    return img_arr

def preprocess_image_tta(image: Image.Image) -> np.ndarray:
    """
    Creates a calibrated multi-view batch (TTA) for robust inference:
    1. Standard full resize to (224, 224)
    2. Aspect-ratio preserving central crop to (224, 224) to capture centered lesions without squashing
    3. Horizontal mirror of standard resize
    4. Horizontal mirror of central crop
    Returns a batch of shape (4, 224, 224, 3) float32 in [0, 255].
    """
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL Image, got {type(image)}")

    rgb = image.convert('RGB')
    
    # 1. Standard resize
    v1 = np.asarray(rgb.resize(IMG_SIZE, Image.Resampling.BILINEAR), dtype=np.float32)
    
    # 2. Central crop preserving aspect ratio
    w, h = rgb.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    cropped = rgb.crop((left, top, left + min_dim, top + min_dim))
    v2 = np.asarray(cropped.resize(IMG_SIZE, Image.Resampling.BILINEAR), dtype=np.float32)
    
    # 3. Horizontal flip of full image
    v3 = np.asarray(rgb.transpose(Image.FLIP_LEFT_RIGHT).resize(IMG_SIZE, Image.Resampling.BILINEAR), dtype=np.float32)
    
    # 4. Horizontal flip of central crop
    v4 = np.asarray(cropped.transpose(Image.FLIP_LEFT_RIGHT).resize(IMG_SIZE, Image.Resampling.BILINEAR), dtype=np.float32)

    return np.stack([v1, v2, v3, v4], axis=0)

