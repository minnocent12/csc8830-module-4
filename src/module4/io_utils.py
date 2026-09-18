"""Safe image and canonical-mask IO for Module 4.

OpenCV decodes color images in BGR order. The core foundation preserves that convention and
leaves conversion to RGB for display code. Binary masks are canonicalized internally as
boolean arrays: False is background and True is foreground.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from module4.types import ColorOrder, ImageMetadata

SUPPORTED_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"})


def validate_image_array(
    image: np.ndarray,
    *,
    name: str = "image",
    color_order: ColorOrder | None = None,
) -> np.ndarray:
    """Validate a non-empty 2D or 3D image array and return it unchanged.

    The function never mutates the input. Channel counts are restricted to common OpenCV
    grayscale, BGR/RGB, and four-channel image representations.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if image.ndim not in (2, 3):
        raise ValueError(f"{name} must be 2D or 3D, got {image.ndim}D")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError(f"{name} must have positive height and width")
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError(f"{name} must have 1, 3, or 4 channels, got {image.shape[2]}")
    if image.dtype.kind in "fc" and not np.isfinite(image).all():
        raise ValueError(f"{name} contains non-finite values")
    if color_order == "GRAY" and image.ndim == 3 and image.shape[2] != 1:
        raise ValueError("GRAY images must be 2D or have one channel")
    if color_order in ("BGR", "RGB") and image.ndim == 2:
        raise ValueError(f"{color_order} images must have color channels")
    return image


def decode_image_bgr(data: bytes, *, source_name: str | None = None) -> np.ndarray:
    """Decode uploaded bytes as a copied OpenCV BGR image."""
    if not data:
        raise ValueError("image data is empty")
    encoded = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        label = f" from {source_name}" if source_name else ""
        raise ValueError(f"could not decode color image{label}")
    return validate_image_array(image, name="decoded BGR image", color_order="BGR").copy()


def decode_image_unchanged(data: bytes, *, source_name: str | None = None) -> np.ndarray:
    """Decode uploaded bytes while preserving grayscale channels and source bit depth."""
    if not data:
        raise ValueError("image data is empty")
    encoded = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        label = f" from {source_name}" if source_name else ""
        raise ValueError(f"could not decode image{label}")
    return validate_image_array(image, name="decoded unchanged image").copy()


def load_image_bgr(path: str | Path) -> np.ndarray:
    """Load a path as a BGR uint8 image without modifying the source file."""
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode color image: {image_path}")
    return validate_image_array(image, name="loaded BGR image", color_order="BGR").copy()


def load_image_unchanged(path: str | Path) -> np.ndarray:
    """Load an image while preserving its source bit depth and channel count."""
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"could not decode image: {image_path}")
    return validate_image_array(image, name="loaded unchanged image").copy()


def normalize_binary_mask(mask: np.ndarray, *, name: str = "mask") -> np.ndarray:
    """Return a new boolean mask with False background and True foreground.

    Any finite nonzero numeric value is treated as foreground. This supports OpenCV masks
    encoded as 0/255 while keeping one canonical representation for later metrics.
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if mask.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got {mask.ndim}D")
    if mask.size == 0:
        raise ValueError(f"{name} must not be empty")
    if mask.dtype.kind in "fc" and not np.isfinite(mask).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.asarray(mask != 0, dtype=bool).copy()


def mask_to_uint8(mask: np.ndarray, *, name: str = "mask") -> np.ndarray:
    """Convert a binary mask to a new OpenCV display/export array containing 0 or 255."""
    normalized = normalize_binary_mask(mask, name=name)
    return np.where(normalized, 255, 0).astype(np.uint8, copy=False)


def describe_image(
    image: np.ndarray,
    *,
    color_order: ColorOrder,
    source_name: str | None = None,
) -> ImageMetadata:
    """Return shape/dtype metadata after validating the image."""
    validated = validate_image_array(image, color_order=color_order)
    channels = 1 if validated.ndim == 2 else validated.shape[2]
    return ImageMetadata(
        height=validated.shape[0],
        width=validated.shape[1],
        channels=channels,
        dtype=str(validated.dtype),
        color_order=color_order,
        source_name=source_name,
    )
