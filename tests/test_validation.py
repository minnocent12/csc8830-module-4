from __future__ import annotations

import numpy as np
import pytest

from module4.validation import (
    ReferenceValidationError,
    align_reference_mask,
    canonicalize_mask,
    prepare_reference,
    validate_reference_identity,
    validate_reference_orientation,
)


def test_canonicalize_mask_accepts_bool_zero_one_and_255() -> None:
    source = np.array([[False, True], [0, 255]], dtype=np.uint8)
    original = source.copy()

    result = canonicalize_mask(source)

    assert result.dtype == bool
    np.testing.assert_array_equal(result, [[False, True], [False, True]])
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize(
    "mask, error",
    [
        (np.array([[0, 2]], dtype=np.uint8), "limited to 0, 1, and 255"),
        (np.array([[0.0, 1.0]], dtype=np.float32), "bool or uint8"),
        (np.zeros((2, 2, 1), dtype=np.uint8), "2D"),
    ],
)
def test_canonicalize_mask_rejects_ambiguous_external_values(mask: np.ndarray, error: str) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        canonicalize_mask(mask)


def test_prepare_reference_keeps_pending_references_without_metrics() -> None:
    result = prepare_reference(
        None,
        expected_shape=(4, 5),
        reference_status="pending",
        reference_type="ground_truth",
        image_id="frame-01",
    )

    assert result.status == "pending"
    assert result.mask is None
    assert result.alignment is None
    assert "pending" in result.warnings[0]


def test_prepare_reference_requires_exact_shape_and_matching_metadata() -> None:
    reference = np.zeros((3, 4), dtype=bool)
    with pytest.raises(ReferenceValidationError, match="does not match"):
        prepare_reference(
            reference,
            expected_shape=(4, 4),
            reference_status="available",
            reference_type="user_reference",
            image_id="frame-01",
            reference_image_id="frame-01",
        )
    with pytest.raises(ReferenceValidationError, match="identities"):
        validate_reference_identity(prediction_image_id="frame-01", reference_image_id="frame-02")
    with pytest.raises(ReferenceValidationError, match="orientations"):
        validate_reference_orientation(prediction_orientation="original", reference_orientation="rotated")


def test_explicit_alignment_is_nearest_neighbor_and_records_metadata() -> None:
    source = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    original = source.copy()

    result, metadata = align_reference_mask(
        source,
        (4, 4),
        prediction_image_id="frame-01",
        reference_image_id="frame-01",
        reference_type="ground_truth",
    )

    expected = np.repeat(np.repeat(source.astype(bool), 2, axis=0), 2, axis=1)
    np.testing.assert_array_equal(result, expected)
    assert metadata.original_dimensions == (2, 2)
    assert metadata.destination_dimensions == (4, 4)
    assert metadata.transformation == "resize_nearest_neighbor"
    assert metadata.interpolation == "nearest_neighbor"
    assert metadata.occurred is True
    np.testing.assert_array_equal(source, original)


def test_same_shape_alignment_is_an_explicit_no_op() -> None:
    result, metadata = align_reference_mask(
        np.array([[0, 255]], dtype=np.uint8),
        (1, 2),
        prediction_image_id="frame-01",
        reference_image_id="frame-01",
        reference_type="sam2_reference",
    )

    assert result.dtype == bool
    assert metadata.transformation == "none"
    assert metadata.interpolation == "not_applicable"
    assert metadata.occurred is False


def test_alignment_also_rejects_explicit_orientation_mismatch() -> None:
    with pytest.raises(ReferenceValidationError, match="orientations"):
        align_reference_mask(
            np.zeros((2, 2), dtype=bool),
            (4, 4),
            prediction_image_id="frame-01",
            reference_image_id="frame-01",
            reference_type="ground_truth",
            prediction_orientation="original",
            reference_orientation="rotated",
        )
