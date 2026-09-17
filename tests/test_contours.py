from __future__ import annotations

import cv2
import numpy as np

from module4.contours import draw_contours_on_bgr, draw_roi_on_bgr, extract_external_contours
from module4.types import ROI


def test_external_contour_does_not_mutate_mask() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 6:14] = True
    original = mask.copy()

    contours = extract_external_contours(mask)

    assert len(contours) == 1
    assert cv2.contourArea(contours[0]) > 0
    np.testing.assert_array_equal(mask, original)


def test_boundary_and_roi_overlays_copy_the_input_image() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 6:14] = True
    contours = extract_external_contours(mask)
    original = image.copy()

    boundary = draw_contours_on_bgr(image, contours)
    roi_overlay = draw_roi_on_bgr(image, ROI(4, 4, 12, 12))

    np.testing.assert_array_equal(image, original)
    assert boundary.shape == image.shape
    assert roi_overlay.shape == image.shape
    assert np.any(boundary != image)
    assert np.any(roi_overlay != image)
