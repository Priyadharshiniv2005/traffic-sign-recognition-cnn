import os
import cv2
import numpy as np

IMG_HEIGHT = 32
IMG_WIDTH = 32

class PreprocessingError(Exception):
    pass

def load_image(image_path):
    """Load image from file path with error handling."""
    if not os.path.exists(image_path):
        raise PreprocessingError(f"Image not found: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise PreprocessingError(f"Failed to load image: {image_path}")
    return img


def resize_image(img, height=IMG_HEIGHT, width=IMG_WIDTH):
    """Resize image to target dimensions."""
    try:
        return cv2.resize(img, (width, height))
    except Exception as e:
        raise PreprocessingError(f"Resize failed: {e}")


def grayscale_to_rgb_if_needed(img):
    """Ensure image has 3 channels."""
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if len(img.shape) == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def normalize_image(img):
    """Normalize pixel values to [0,1]."""
    img = img.astype(np.float32) / 255.0
    return img


def preprocess_single_image(image_path):
    """Preprocess a single image for prediction."""
    img = load_image(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = resize_image(img)
    img = normalize_image(img)
    img = np.expand_dims(img, axis=0)
    return img


def preprocess_for_training(X):
    """Normalize training data."""
    X = X.astype(np.float32) / 255.0
    return X

