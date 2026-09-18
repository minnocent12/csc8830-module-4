from __future__ import annotations

import cv2
import numpy as np
import pytest

from module4.io_utils import (
    decode_image_bgr,
    decode_image_unchanged,
    describe_image,
    mask_to_uint8,
    normalize_binary_mask,
    validate_image_array,
)


def test_validate_image_array_preserves_pixels() -> None:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    image[1, 2] = (1, 2, 3)
    original = image.copy()

    result = validate_image_array(image, color_order="BGR")

    assert result is image
    np.testing.assert_array_equal(image, original)


def test_validate_image_array_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="2D or 3D"):
        validate_image_array(np.zeros((2, 3, 4, 1), dtype=np.uint8))


def test_decode_image_bgr_returns_expected_shape_and_order() -> None:
    source = np.zeros((3, 4, 3), dtype=np.uint8)
    source[1, 2] = (10, 20, 30)
    success, encoded = cv2.imencode(".png", source)
    assert success

    decoded = decode_image_bgr(encoded.tobytes())

    assert decoded.shape == source.shape
    assert decoded.dtype == np.uint8
    np.testing.assert_array_equal(decoded, source)


def test_decode_image_bgr_rejects_empty_bytes() -> None:
    with pytest.raises(ValueError, match="empty"):
        decode_image_bgr(b"")


def test_decode_image_unchanged_preserves_grayscale_bit_depth() -> None:
    source = np.array([[0, 1024], [50000, 65535]], dtype=np.uint16)
    success, encoded = cv2.imencode(".png", source)
    assert success

    decoded = decode_image_unchanged(encoded.tobytes(), source_name="thermal.png")

    assert decoded.dtype == np.uint16
    np.testing.assert_array_equal(decoded, source)


def test_normalize_binary_mask_uses_false_background_and_copies() -> None:
    source = np.array([[0, 255], [2, 0]], dtype=np.uint8)
    normalized = normalize_binary_mask(source)

    assert normalized.dtype == bool
    np.testing.assert_array_equal(normalized, [[False, True], [True, False]])
    normalized[0, 0] = True
    assert source[0, 0] == 0


def test_mask_to_uint8_exports_zero_or_255() -> None:
    result = mask_to_uint8(np.array([[False, True], [True, False]]))
    np.testing.assert_array_equal(result, [[0, 255], [255, 0]])


def test_mask_validation_rejects_non_2d_and_nonfinite() -> None:
    with pytest.raises(ValueError, match="2D"):
        normalize_binary_mask(np.zeros((2, 2, 1), dtype=np.uint8))
    with pytest.raises(ValueError, match="non-finite"):
        normalize_binary_mask(np.array([[0.0, np.nan]]))


def test_describe_image_records_shape_dtype_and_channel_order() -> None:
    metadata = describe_image(
        np.zeros((7, 8, 3), dtype=np.uint8),
        color_order="BGR",
        source_name="fixture.png",
    )

    assert metadata.height == 7
    assert metadata.width == 8
    assert metadata.channels == 3
    assert metadata.dtype == "uint8"
    assert metadata.color_order == "BGR"
    assert metadata.source_name == "fixture.png"
