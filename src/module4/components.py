"""Classical connected-component selection for RGB masks."""
from __future__ import annotations

import cv2
import numpy as np

from module4.io_utils import mask_to_uint8, normalize_binary_mask
from module4.types import ComponentSelection, ROI


def select_roi_component(
    mask: np.ndarray,
    roi: ROI,
    *,
    minimum_area: int = 1,
) -> ComponentSelection | None:
    """Select the component with the greatest ROI overlap, then greatest area.

    Only components meeting minimum_area and having at least one pixel inside the ROI are
    candidates. The deterministic ranking is (ROI overlap descending, area descending, label
    ascending). A tie on the first two values is reported as ambiguous even though the lowest
    label is selected reproducibly.
    """
    normalized = normalize_binary_mask(mask)
    if not isinstance(minimum_area, (int, np.integer)) or isinstance(minimum_area, bool):
        raise TypeError("minimum_area must be an integer")
    minimum_area = int(minimum_area)
    if minimum_area <= 0:
        raise ValueError("minimum_area must be positive")
    height, width = normalized.shape
    if roi.x < 0 or roi.y < 0 or roi.width <= 0 or roi.height <= 0:
        raise ValueError("ROI must have non-negative origin and positive dimensions")
    if roi.x + roi.width > width or roi.y + roi.height > height:
        raise ValueError("ROI must lie fully inside the mask")

    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_to_uint8(normalized), connectivity=8
    )
    roi_slice = np.zeros_like(normalized, dtype=bool)
    roi_slice[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width] = True
    candidates: list[tuple[int, int, int]] = []
    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum_area:
            continue
        component = labels == label
        overlap = int(np.count_nonzero(component & roi_slice))
        if overlap:
            candidates.append((label, area, overlap))
    if not candidates:
        return None

    candidates.sort(key=lambda item: (-item[2], -item[1], item[0]))
    selected_label, selected_area, selected_overlap = candidates[0]
    ambiguous = sum(
        candidate[1] == selected_area and candidate[2] == selected_overlap
        for candidate in candidates
    ) > 1
    return ComponentSelection(
        label=selected_label,
        area=selected_area,
        roi_overlap=selected_overlap,
        candidate_count=len(candidates),
        ambiguous=ambiguous,
        mask=(labels == selected_label),
    )
