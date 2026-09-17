"""Explicit data structures shared by Module 4 processing code."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np

ColorOrder = Literal["BGR", "RGB", "GRAY", "THERMAL"]
ReferenceStatusValue = Literal["pending", "completed", "failed", "unavailable"]


@dataclass(frozen=True)
class ImageMetadata:
    """Describes an image array without changing or copying its pixels."""

    height: int
    width: int
    channels: int
    dtype: str
    color_order: ColorOrder
    source_name: str | None = None


@dataclass(frozen=True)
class ReferenceStatus:
    """Reports whether an optional comparison reference is available."""

    status: ReferenceStatusValue
    message: str
    source: str | None = None


@dataclass(frozen=True)
class ROI:
    """A strict image rectangle represented as x, y, width, height."""

    x: int
    y: int
    width: int
    height: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        """Return the rectangle in OpenCV's xywh representation."""
        return self.x, self.y, self.width, self.height


@dataclass(frozen=True)
class RGBPipelineConfig:
    """Small, documented set of RGB pipeline parameters."""

    grabcut_iterations: int = 5
    opening_kernel_size: int = 3
    closing_kernel_size: int = 5
    minimum_component_area: int = 16
    contour_thickness: int = 2


@dataclass(frozen=True)
class ComponentSelection:
    """The component selected by the classical ROI-overlap rule."""

    label: int
    area: int
    roi_overlap: int
    candidate_count: int
    ambiguous: bool
    mask: np.ndarray


@dataclass(frozen=True)
class RGBSegmentationResult:
    """All meaningful outputs of the ROI-assisted RGB pipeline."""

    original_bgr: np.ndarray
    roi: ROI
    roi_overlay_bgr: np.ndarray
    grayscale: np.ndarray
    hsv: np.ndarray
    lab: np.ndarray
    grabcut_labels: np.ndarray
    raw_foreground_mask: np.ndarray
    cleaned_foreground_mask: np.ndarray
    selected_component_mask: np.ndarray
    final_mask: np.ndarray
    contours: tuple[np.ndarray, ...]
    boundary_overlay_bgr: np.ndarray
    parameters: Mapping[str, int]
    warnings: tuple[str, ...]
    component_selection: ComponentSelection | None
