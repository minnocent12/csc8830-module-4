from __future__ import annotations

import cv2
import numpy as np
import pytest

from module4.preprocessing import (
    apply_morphological_cleanup,
    grabcut_foreground_mask,
    validate_kernel_size,
)


def test_grabcut_label_conversion_keeps_only_foreground_labels() -> None:
    labels = np.array(
        [[cv2.GC_BGD, cv2.GC_FGD], [cv2.GC_PR_BGD, cv2.GC_PR_FGD]],
        dtype=np.uint8,
    )

    result = grabcut_foreground_mask(labels)

    np.testing.assert_array_equal(result, [[False, True], [False, True]])
    np.testing.assert_array_equal(labels, [[0, 1], [2, 3]])


def test_grabcut_label_conversion_rejects_unknown_labels() -> None:
    with pytest.raises(ValueError, match="unknown"):
        grabcut_foreground_mask(np.array([[99]], dtype=np.uint8))


def test_kernel_size_requires_positive_odd_integer() -> None:
    assert validate_kernel_size(3) == 3
    with pytest.raises(ValueError):
        validate_kernel_size(2)
    with pytest.raises(ValueError):
        validate_kernel_size(0)
    with pytest.raises(TypeError):
        validate_kernel_size(3.0)  # type: ignore[arg-type]


def test_opening_removes_isolated_pixel_and_preserves_large_region() -> None:
    mask = np.zeros((11, 11), dtype=bool)
    mask[1, 1] = True
    mask[4:9, 4:9] = True

    cleaned = apply_morphological_cleanup(
        mask,
        opening_kernel_size=3,
        closing_kernel_size=1,
    )

    assert not cleaned[1, 1]
    assert cleaned[6, 6]
