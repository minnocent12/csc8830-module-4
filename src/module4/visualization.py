"""Display-boundary helpers for Module 4 images."""
from __future__ import annotations

import cv2
import numpy as np

from module4.contours import draw_contours_on_bgr, extract_external_contours
from module4.io_utils import mask_to_uint8, validate_image_array
from module4.validation import validate_mask_pair


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


def mask_overlap_bgr(prediction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Render TP, FP, and FN pixels as a new BGR comparison image.

    True positives are green, false positives are red, false negatives are blue, and shared
    background is black. The inputs are canonicalized and shape-checked without mutation.
    """
    prediction_bool, reference_bool = validate_mask_pair(prediction, reference)
    overlap = np.zeros((*prediction_bool.shape, 3), dtype=np.uint8)
    true_positive = prediction_bool & reference_bool
    false_positive = prediction_bool & ~reference_bool
    false_negative = ~prediction_bool & reference_bool
    overlap[true_positive] = (0, 200, 0)
    overlap[false_positive] = (0, 0, 255)
    overlap[false_negative] = (255, 0, 0)
    return overlap


# BGR boundary colors shared by the web app and the report figures.
CLASSICAL_BOUNDARY_BGR = (0, 0, 255)  # red
SAM2_BOUNDARY_BGR = (255, 255, 0)  # cyan


def boundary_comparison_bgr(
    image_bgr: np.ndarray,
    classical_mask: np.ndarray,
    sam2_mask: np.ndarray,
    *,
    thickness: int = 2,
) -> np.ndarray:
    """Draw the classical (red) and SAM2 (cyan) external boundaries on one BGR copy.

    Both masks are shape-checked against each other; contours come from
    ``extract_external_contours`` exactly as for the single-method boundary overlays.
    """
    classical_bool, sam2_bool = validate_mask_pair(classical_mask, sam2_mask)
    if classical_bool.shape != image_bgr.shape[:2]:
        raise ValueError("boundary comparison masks must match the image height and width")
    overlay = draw_contours_on_bgr(
        image_bgr, extract_external_contours(classical_bool), color=CLASSICAL_BOUNDARY_BGR, thickness=thickness
    )
    return draw_contours_on_bgr(
        overlay, extract_external_contours(sam2_bool), color=SAM2_BOUNDARY_BGR, thickness=thickness
    )
