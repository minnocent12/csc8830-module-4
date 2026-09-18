"""Conservative binary-mask and reference-readiness validation for evaluation."""
from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from module4.types import (
    AlignmentMetadata,
    EvaluationReferenceStatus,
    ReferenceType,
    ReferenceValidation,
)


class ReferenceValidationError(ValueError):
    """Raised when a reference cannot be safely compared with a prediction."""


def canonicalize_mask(mask: np.ndarray, *, name: str = "mask") -> np.ndarray:
    """Convert explicitly supported external binary encodings to a copied bool mask.

    Accepted encodings are bool and uint8 containing only 0, 1, and/or 255. Arbitrary grayscale
    values are rejected because this function does not perform an implicit threshold operation.
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if mask.ndim != 2:
        raise ValueError(f"{name} must be a 2D binary mask")
    if mask.size == 0:
        raise ValueError(f"{name} must not be empty")
    if mask.dtype == np.bool_:
        return mask.copy()
    if mask.dtype != np.uint8:
        raise TypeError(f"{name} must use bool or uint8 binary encoding")
    values = np.unique(mask)
    if not np.isin(values, np.array([0, 1, 255], dtype=np.uint8)).all():
        raise ValueError(f"{name} uint8 values must be limited to 0, 1, and 255")
    return (mask != 0).astype(bool, copy=True)


def validate_mask_pair(prediction: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Canonicalize two masks and require identical height and width without resizing."""
    prediction_bool = canonicalize_mask(prediction, name="prediction mask")
    reference_bool = canonicalize_mask(reference, name="reference mask")
    if prediction_bool.shape != reference_bool.shape:
        raise ValueError(
            "prediction and reference masks must have the same height and width; "
            "resize/alignment is never implicit"
        )
    return prediction_bool, reference_bool


def _validate_reference_type(reference_type: ReferenceType | None) -> ReferenceType | None:
    allowed = {"ground_truth", "sam2_reference", "user_reference"}
    if reference_type is not None and reference_type not in allowed:
        raise ValueError(f"reference_type must be one of {sorted(allowed)}")
    return reference_type


def _validate_dimensions(dimensions: Sequence[int], *, name: str) -> tuple[int, int]:
    values = tuple(dimensions)
    if len(values) != 2 or not all(isinstance(value, (int, np.integer)) for value in values):
        raise ValueError(f"{name} must contain height and width integers")
    height, width = (int(value) for value in values)
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} must contain positive dimensions")
    return height, width


def validate_reference_identity(
    *,
    prediction_image_id: str | None,
    reference_image_id: str | None,
) -> None:
    """Reject an explicitly mismatched source identity."""
    if prediction_image_id is not None and reference_image_id is not None:
        if prediction_image_id != reference_image_id:
            raise ReferenceValidationError(
                "prediction and reference image identities do not match; numerical comparison refused"
            )


def validate_reference_orientation(
    *,
    prediction_orientation: str | None,
    reference_orientation: str | None,
) -> None:
    """Reject an explicitly mismatched orientation."""
    if prediction_orientation is not None and reference_orientation is not None:
        if prediction_orientation != reference_orientation:
            raise ReferenceValidationError(
                "prediction and reference orientations do not match; numerical comparison refused"
            )


def align_reference_mask(
    reference_mask: np.ndarray,
    destination_shape: Sequence[int],
    *,
    prediction_image_id: str,
    reference_image_id: str,
    reference_type: ReferenceType,
    prediction_orientation: str | None = None,
    reference_orientation: str | None = None,
) -> tuple[np.ndarray, AlignmentMetadata]:
    """Explicitly align a binary reference with nearest-neighbor interpolation only."""
    validate_reference_identity(
        prediction_image_id=prediction_image_id,
        reference_image_id=reference_image_id,
    )
    validate_reference_orientation(
        prediction_orientation=prediction_orientation,
        reference_orientation=reference_orientation,
    )
    _validate_reference_type(reference_type)
    canonical = canonicalize_mask(reference_mask, name="reference mask")
    destination = _validate_dimensions(destination_shape, name="destination_shape")
    original = (int(canonical.shape[0]), int(canonical.shape[1]))
    if original == destination:
        alignment = AlignmentMetadata(original, destination, "none", "not_applicable", False)
        return canonical, alignment
    resized = cv2.resize(
        canonical.astype(np.uint8),
        (destination[1], destination[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    alignment = AlignmentMetadata(
        original,
        destination,
        "resize_nearest_neighbor",
        "nearest_neighbor",
        True,
    )
    return resized.astype(bool), alignment


def prepare_reference(
    reference_mask: np.ndarray | None,
    *,
    expected_shape: Sequence[int],
    reference_status: EvaluationReferenceStatus,
    reference_type: ReferenceType | None,
    image_id: str | None = None,
    reference_image_id: str | None = None,
    prediction_orientation: str | None = None,
    reference_orientation: str | None = None,
) -> ReferenceValidation:
    """Validate reference availability, identity, orientation, and exact dimensions."""
    _validate_reference_type(reference_type)
    expected = _validate_dimensions(expected_shape, name="expected_shape")
    if reference_status == "pending":
        return ReferenceValidation(
            "pending", reference_type, reference_image_id, None, None,
            ("Reference mask is pending; metrics are unavailable until it is supplied.",),
        )
    if reference_status == "failed":
        return ReferenceValidation("failed", reference_type, reference_image_id, None, None, ())
    if reference_status != "available":
        raise ValueError("reference_status must be available, pending, or failed")
    if reference_mask is None:
        raise ReferenceValidationError("reference status is available but no reference mask was supplied")
    validate_reference_identity(
        prediction_image_id=image_id,
        reference_image_id=reference_image_id,
    )
    validate_reference_orientation(
        prediction_orientation=prediction_orientation,
        reference_orientation=reference_orientation,
    )
    canonical = canonicalize_mask(reference_mask, name="reference mask")
    if canonical.shape != expected:
        raise ReferenceValidationError(
            f"reference mask shape {canonical.shape} does not match prediction shape {expected}; "
            "invoke align_reference_mask explicitly if alignment is justified"
        )
    return ReferenceValidation(
        "available",
        reference_type,
        reference_image_id,
        canonical,
        AlignmentMetadata(expected, expected, "none", "not_applicable", False),
        (),
    )
