from __future__ import annotations

import cv2
import numpy as np
import pytest

import module4.reference.sam2_reference as sam2_reference
from module4.reference import (
    bgr_to_sam2_rgb,
    check_sam2_availability,
    run_sam2_reference,
    select_sam2_mask,
    thermal_to_sam2_rgb,
    validate_box_prompt,
)
from module4.types import SAM2Config, SAM2Prompt
from module4.validation import ReferenceValidationError


def test_bgr_conversion_is_explicit_and_does_not_mutate_source() -> None:
    source = np.array([[[1, 2, 3], [10, 20, 30]]], dtype=np.uint8)
    original = source.copy()

    converted = bgr_to_sam2_rgb(source)

    np.testing.assert_array_equal(converted, source[:, :, ::-1])
    np.testing.assert_array_equal(source, original)
    assert converted.dtype == np.uint8


def test_thermal_conversion_renders_grayscale_as_three_equal_rgb_channels() -> None:
    source = np.array([[0, 10], [100, 200]], dtype=np.uint16)

    converted = thermal_to_sam2_rgb(source)

    assert converted.shape == (2, 2, 3)
    assert converted.dtype == np.uint8
    np.testing.assert_array_equal(converted[:, :, 0], converted[:, :, 1])
    np.testing.assert_array_equal(converted[:, :, 1], converted[:, :, 2])


def test_false_color_thermal_conversion_is_only_a_channel_order_conversion() -> None:
    source = np.array([[[4, 5, 6], [20, 30, 40]]], dtype=np.uint8)
    original = source.copy()

    converted = thermal_to_sam2_rgb(source)

    np.testing.assert_array_equal(converted, cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
    np.testing.assert_array_equal(source, original)


def test_box_prompt_validation_returns_normalized_prompt() -> None:
    prompt = validate_box_prompt(SAM2Prompt(1, 2, 3, 4), (10, 12, 3))

    assert prompt == SAM2Prompt(1, 2, 3, 4)
    assert prompt.as_xyxy() == (1, 2, 4, 6)


@pytest.mark.parametrize(
    "prompt",
    [SAM2Prompt(-1, 0, 2, 2), SAM2Prompt(0, 0, 0, 2), SAM2Prompt(9, 0, 2, 2)],
)
def test_box_prompt_validation_rejects_invalid_boxes(prompt: SAM2Prompt) -> None:
    with pytest.raises((TypeError, ValueError)):
        validate_box_prompt(prompt, (10, 10, 3))


def test_native_score_selection_is_deterministic_and_does_not_mutate_masks() -> None:
    masks = np.array(
        [
            [[0, 1], [1, 0]],
            [[1, 1], [0, 0]],
            [[0, 0], [1, 1]],
        ],
        dtype=np.uint8,
    )
    original = masks.copy()

    selected, index, scores = select_sam2_mask(masks, np.array([0.6, 0.9, 0.9]), expected_shape=(2, 2))

    assert index == 1
    assert scores == (0.6, 0.9, 0.9)
    np.testing.assert_array_equal(selected, masks[1].astype(bool))
    np.testing.assert_array_equal(masks, original)


def test_selection_rejects_logits_and_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="binary"):
        select_sam2_mask(np.array([[[0.2, 0.8], [0.1, 0.9]]]), np.array([0.5]), expected_shape=(2, 2))
    with pytest.raises(ReferenceValidationError, match="shape"):
        select_sam2_mask(np.ones((1, 2, 3), dtype=np.uint8), np.array([0.5]), expected_shape=(2, 2))


def test_unavailable_and_pending_states_have_no_placeholder_mask(tmp_path) -> None:
    unavailable = check_sam2_availability(SAM2Config(checkpoint_path=str(tmp_path / "missing.pt")))
    assert unavailable.status == "unavailable"
    assert unavailable.mask is None
    assert unavailable.error is not None

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    result = run_sam2_reference(image, SAM2Prompt(1, 1, 4, 4), source_image_id="frame-1")
    assert result.status == "unavailable"
    assert result.mask is None
    assert result.source_dimensions == (8, 8)
    assert result.prompt == SAM2Prompt(1, 1, 4, 4)


def test_missing_optional_dependency_is_reported_without_importing_it(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "sam2.1_hiera_tiny.pt"
    checkpoint.write_bytes(b"not a model")
    monkeypatch.setattr(
        sam2_reference,
        "_load_sam2_dependencies",
        lambda: (None, None, "official SAM2 package is unavailable"),
    )

    result = check_sam2_availability(SAM2Config(checkpoint_path=str(checkpoint)))

    assert result.status == "unavailable"
    assert result.mask is None
    assert result.error == "official SAM2 package is unavailable"
