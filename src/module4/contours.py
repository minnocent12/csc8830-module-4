"""Contour extraction and non-destructive BGR boundary rendering."""
from __future__ import annotations

import cv2
import numpy as np

from module4.io_utils import mask_to_uint8, normalize_binary_mask, validate_image_array
from module4.types import ROI


def extract_external_contours(mask: np.ndarray) -> tuple[np.ndarray, ...]:
    """Extract external contours from a copied 0/255 mask.

    RETR_EXTERNAL is used because the assignment asks for the outer human boundary. OpenCV is
    allowed to modify its input during contour extraction, so a private copy is passed.
    """
    binary = mask_to_uint8(normalize_binary_mask(mask))
    contours, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return tuple(contour.copy() for contour in contours)


def draw_contours_on_bgr(
    image_bgr: np.ndarray,
    contours: tuple[np.ndarray, ...],
    *,
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw contours on a copy of a validated BGR image."""
    validate_image_array(image_bgr, name="BGR image", color_order="BGR")
    if image_bgr.dtype != np.uint8 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("boundary rendering requires an H x W x 3 uint8 BGR image")
    if thickness <= 0:
        raise ValueError("contour thickness must be positive")
    if len(color) != 3 or any(channel < 0 or channel > 255 for channel in color):
        raise ValueError("contour color must contain three values in [0, 255]")
    overlay = image_bgr.copy()
    if contours:
        cv2.drawContours(overlay, list(contours), -1, color, int(thickness))
    return overlay


def draw_roi_on_bgr(
    image_bgr: np.ndarray,
    roi: ROI,
    *,
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw an ROI rectangle on a copy of a BGR image."""
    validate_image_array(image_bgr, name="BGR image", color_order="BGR")
    if image_bgr.dtype != np.uint8 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("ROI rendering requires an H x W x 3 uint8 BGR image")
    if thickness <= 0:
        raise ValueError("ROI thickness must be positive")
    overlay = image_bgr.copy()
    start = (roi.x, roi.y)
    end = (roi.x + roi.width - 1, roi.y + roi.height - 1)
    cv2.rectangle(overlay, start, end, color, int(thickness))
    return overlay
