"""Classical RGB preprocessing and mask-cleanup helpers.

These helpers deliberately expose the OpenCV operations used by the RGB pipeline. They do not
detect a person and contain no learned or pretrained component.
"""
from __future__ import annotations

import cv2
import numpy as np

from module4.io_utils import mask_to_uint8, normalize_binary_mask, validate_image_array


def validate_kernel_size(size: int, *, name: str = "kernel size") -> int:
    """Validate an odd positive morphology kernel size."""
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    size = int(size)
    if size <= 0 or size % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer")
    return size


def bgr_diagnostics(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return grayscale, HSV, and Lab diagnostics from a BGR uint8 image.

    OpenCV expects the input in BGR order. The returned arrays are diagnostics only; GrabCut
    remains the primary RGB operation in the pipeline.
    """
    validate_image_array(image_bgr, name="RGB BGR image", color_order="BGR")
    if image_bgr.dtype != np.uint8 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("RGB processing requires an H x W x 3 uint8 BGR image")
    return (
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV),
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB),
    )


def grabcut_foreground_mask(labels: np.ndarray) -> np.ndarray:
    """Convert GrabCut labels to the canonical boolean foreground mask.

    Definite foreground (GC_FGD) and probable foreground (GC_PR_FGD) are foreground. Definite
    background and probable background remain background. The label array is not modified.
    """
    if not isinstance(labels, np.ndarray) or labels.ndim != 2:
        raise ValueError("GrabCut labels must be a 2D numpy array")
    valid_labels = {
        cv2.GC_BGD,
        cv2.GC_FGD,
        cv2.GC_PR_BGD,
        cv2.GC_PR_FGD,
    }
    if not np.isin(labels, tuple(valid_labels)).all():
        raise ValueError("GrabCut labels contain an unknown label")
    return np.logical_or(labels == cv2.GC_FGD, labels == cv2.GC_PR_FGD).copy()


def apply_morphological_cleanup(
    mask: np.ndarray,
    *,
    opening_kernel_size: int = 3,
    closing_kernel_size: int = 5,
) -> np.ndarray:
    """Apply documented opening then closing to a binary mask.

    Opening removes small isolated foreground specks. Closing fills small holes and joins
    nearby gaps. The result uses the canonical boolean mask convention and the input is not
    modified.
    """
    normalized = normalize_binary_mask(mask)
    opening_size = validate_kernel_size(opening_kernel_size, name="opening kernel size")
    closing_size = validate_kernel_size(closing_kernel_size, name="closing kernel size")
    cleaned = mask_to_uint8(normalized)
    if opening_size > 1:
        opening_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (opening_size, opening_size)
        )
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, opening_kernel)
    if closing_size > 1:
        closing_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (closing_size, closing_size)
        )
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, closing_kernel)
    return cleaned > 0


def normalize_thermal_intensity(image: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Normalize a finite single-channel intensity array to uint8 in ``[0, 255]``.

    The source array is not modified. Supported source dtypes are uint8, uint16, float32,
    and float64. A constant image maps to all zeros and returns equal minimum and maximum
    values instead of dividing by zero.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("thermal intensity must be a numpy.ndarray")
    if image.ndim != 2:
        raise ValueError("thermal intensity must be a 2D array")
    if image.size == 0 or image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("thermal intensity must not be empty")
    supported_dtypes = (np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.float32), np.dtype(np.float64))
    if image.dtype not in supported_dtypes:
        raise TypeError("thermal intensity dtype must be uint8, uint16, float32, or float64")
    if image.dtype.kind == "f" and not np.isfinite(image).all():
        raise ValueError("thermal intensity contains non-finite values")
    values = image.astype(np.float64, copy=False)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if minimum == maximum:
        return np.zeros(image.shape, dtype=np.uint8), minimum, maximum
    normalized = (values - minimum) * (255.0 / (maximum - minimum))
    return np.clip(np.rint(normalized), 0, 255).astype(np.uint8), minimum, maximum
