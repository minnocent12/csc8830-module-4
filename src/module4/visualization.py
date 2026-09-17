"""Display-boundary helpers for Module 4 images."""
from __future__ import annotations

import cv2
import numpy as np

from module4.io_utils import mask_to_uint8, validate_image_array


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR image to a new RGB display array."""
    validate_image_array(image_bgr, name="BGR image", color_order="BGR")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("BGR display conversion requires three channels")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def to_display_uint8(image: np.ndarray) -> np.ndarray:
    """Return a display-safe uint8 copy, clipping numeric values to [0, 255]."""
    validate_image_array(image, name="display image")
    if image.dtype == np.uint8:
        return image.copy()
    return np.clip(image, 0, 255).astype(np.uint8)


def display_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a canonical mask to a display-safe 0/255 array."""
    return mask_to_uint8(mask)
