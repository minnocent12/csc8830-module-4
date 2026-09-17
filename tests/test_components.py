from __future__ import annotations

import numpy as np
import pytest

from module4.components import select_roi_component
from module4.types import ROI


def test_component_selection_prioritizes_roi_overlap_then_area() -> None:
    mask = np.zeros((12, 16), dtype=bool)
    mask[2:5, 2:5] = True
    mask[7:11, 8:14] = True
    roi = ROI(1, 1, 5, 5)

    selected = select_roi_component(mask, roi, minimum_area=2)

    assert selected is not None
    assert selected.area == 9
    assert selected.roi_overlap == 9
    assert selected.candidate_count == 1
    np.testing.assert_array_equal(selected.mask, mask & np.pad(
        np.ones((3, 3), dtype=bool),
        ((2, 7), (2, 11)),
    ))


def test_component_selection_reports_tied_candidates_as_ambiguous() -> None:
    mask = np.zeros((8, 12), dtype=bool)
    mask[1:3, 1:3] = True
    mask[5:7, 8:10] = True

    selected = select_roi_component(mask, ROI(0, 0, 12, 8), minimum_area=1)

    assert selected is not None
    assert selected.candidate_count == 2
    assert selected.ambiguous
    assert selected.label == 1


def test_component_selection_returns_none_for_empty_mask() -> None:
    result = select_roi_component(np.zeros((5, 5), dtype=bool), ROI(0, 0, 5, 5))
    assert result is None


def test_component_selection_rejects_invalid_area() -> None:
    with pytest.raises(ValueError, match="positive"):
        select_roi_component(np.ones((4, 4), dtype=bool), ROI(0, 0, 4, 4), minimum_area=0)
