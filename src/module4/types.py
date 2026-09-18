"""Explicit data structures shared by Module 4 processing code."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np

ColorOrder = Literal["BGR", "RGB", "GRAY", "THERMAL"]
ReferenceStatusValue = Literal["pending", "completed", "failed", "unavailable"]
ThermalPolarity = Literal["bright", "dark"]
ThermalStatus = Literal["selected", "ambiguous", "weak", "constant", "empty"]
ReferenceType = Literal["ground_truth", "sam2_reference", "user_reference"]
EvaluationReferenceStatus = Literal["available", "pending", "failed"]


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


@dataclass(frozen=True)
class ThermalPipelineConfig:
    """Explicit parameters for the classical thermal pipeline."""

    gaussian_blur_kernel_size: int = 1
    opening_kernel_size: int = 3
    closing_kernel_size: int = 5
    minimum_component_area: int = 16
    target_relative_area: float = 0.25
    minimum_selection_score: float = 0.35
    polarity_ambiguity_margin: float = 0.05
    contour_thickness: int = 2


@dataclass(frozen=True)
class ThermalComponentSelection:
    """A component ranked by transparent, classical geometry criteria."""

    polarity: ThermalPolarity
    label: int
    area: int
    relative_area: float
    roi_overlap: int
    roi_score: float
    area_score: float
    centrality_score: float
    border_score: float
    score: float
    centroid_x: float
    centroid_y: float
    touches_border: bool
    candidate_count: int
    mask: np.ndarray


@dataclass(frozen=True)
class ThermalSegmentationResult:
    """Structured outputs from classical dual-polarity thermal segmentation."""

    original_thermal: np.ndarray
    source_metadata: ImageMetadata
    input_representation: str
    source_intensity: np.ndarray
    intensity_min: float
    intensity_max: float
    normalized_intensity: np.ndarray
    enhanced_intensity: np.ndarray
    bright_otsu_threshold: float
    dark_otsu_threshold: float
    bright_raw_mask: np.ndarray
    dark_raw_mask: np.ndarray
    bright_cleaned_mask: np.ndarray
    dark_cleaned_mask: np.ndarray
    selected_polarity: ThermalPolarity | None
    polarity_ambiguous: bool
    bright_component: ThermalComponentSelection | None
    dark_component: ThermalComponentSelection | None
    selected_component: ThermalComponentSelection | None
    final_mask: np.ndarray
    contours: tuple[np.ndarray, ...]
    source_display_bgr: np.ndarray
    boundary_overlay_bgr: np.ndarray
    status: ThermalStatus
    parameters: Mapping[str, int | float]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ConfusionCounts:
    """Pixel-level binary confusion counts."""

    tp: int
    fp: int
    fn: int
    tn: int


@dataclass(frozen=True)
class SegmentationMetrics:
    """Pixel-level metrics and counts for two aligned canonical masks."""

    iou: float
    dice: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int
    tn: int
    prediction_foreground_pixels: int
    reference_foreground_pixels: int


@dataclass(frozen=True)
class AlignmentMetadata:
    """Records an explicit reference-mask alignment operation."""

    original_dimensions: tuple[int, int]
    destination_dimensions: tuple[int, int]
    transformation: str
    interpolation: str
    occurred: bool


@dataclass(frozen=True)
class ReferenceValidation:
    """Result of checking reference availability and mask readiness."""

    status: EvaluationReferenceStatus
    reference_type: ReferenceType | None
    image_id: str | None
    mask: np.ndarray | None
    alignment: AlignmentMetadata | None
    warnings: tuple[str, ...]
