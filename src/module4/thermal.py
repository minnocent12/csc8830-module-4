"""Classical dual-polarity thermal human-boundary segmentation.

This module deliberately uses only deterministic NumPy/OpenCV operations. It evaluates bright
and dark foreground interpretations independently because thermal scenes do not guarantee that
the human is hotter or brighter than the background.
"""
from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import cv2
import numpy as np

from module4.contours import draw_contours_on_bgr, extract_external_contours
from module4.io_utils import describe_image, validate_image_array
from module4.preprocessing import (
    apply_morphological_cleanup,
    normalize_thermal_intensity,
    validate_kernel_size,
)
from module4.rgb import validate_roi
from module4.types import (
    ROI,
    ThermalComponentSelection,
    ThermalPipelineConfig,
    ThermalPolarity,
    ThermalSegmentationResult,
)

SUPPORTED_THERMAL_DTYPES = frozenset(
    {np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.float32), np.dtype(np.float64)}
)


def _validate_config(config: ThermalPipelineConfig) -> ThermalPipelineConfig:
    if not isinstance(config, ThermalPipelineConfig):
        raise TypeError("config must be a ThermalPipelineConfig")
    for name, value in (
        ("gaussian_blur_kernel_size", config.gaussian_blur_kernel_size),
        ("opening_kernel_size", config.opening_kernel_size),
        ("closing_kernel_size", config.closing_kernel_size),
        ("minimum_component_area", config.minimum_component_area),
        ("contour_thickness", config.contour_thickness),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    validate_kernel_size(config.gaussian_blur_kernel_size, name="Gaussian blur kernel size")
    validate_kernel_size(config.opening_kernel_size, name="opening kernel size")
    validate_kernel_size(config.closing_kernel_size, name="closing kernel size")
    for name, value in (
        ("target_relative_area", config.target_relative_area),
        ("minimum_selection_score", config.minimum_selection_score),
        ("polarity_ambiguity_margin", config.polarity_ambiguity_margin),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError(f"{name} must be numeric")
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0 < config.target_relative_area < 1:
        raise ValueError("target_relative_area must be between 0 and 1")
    if not 0 <= config.minimum_selection_score <= 1:
        raise ValueError("minimum_selection_score must be between 0 and 1")
    if not 0 <= config.polarity_ambiguity_margin <= 1:
        raise ValueError("polarity_ambiguity_margin must be between 0 and 1")
    return config


def _source_to_intensity(source: np.ndarray) -> tuple[np.ndarray, str]:
    """Return a copied single-channel intensity view and an honest representation label."""
    validate_image_array(source, name="thermal image")
    if source.dtype not in SUPPORTED_THERMAL_DTYPES:
        raise TypeError("thermal image dtype must be uint8, uint16, float32, or float64")
    if source.ndim == 2:
        return source.copy(), "single-channel intensity"
    if source.shape[2] == 1:
        return source[:, :, 0].copy(), "single-channel intensity (one-channel array)"
    if source.shape[2] != 3:
        raise ValueError("thermal image must be 2D, one-channel, or a 3-channel false-color image")
    # OpenCV decodes color images as BGR. This luminance conversion is only for intensity
    # processing; it does not convert palette values into calibrated temperature.
    bgr = source.astype(np.float64, copy=False)
    intensity = 0.114 * bgr[:, :, 0] + 0.587 * bgr[:, :, 1] + 0.299 * bgr[:, :, 2]
    return intensity, "three-channel false-color BGR palette converted to intensity"


def _source_display_bgr(source: np.ndarray, normalized_intensity: np.ndarray) -> np.ndarray:
    """Build a display-only BGR copy without changing the computational representation."""
    if source.ndim == 2 or source.shape[2] == 1:
        return cv2.cvtColor(normalized_intensity, cv2.COLOR_GRAY2BGR)
    if source.dtype == np.uint8:
        return source.copy()
    values = source.astype(np.float64, copy=False)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if minimum == maximum:
        display = np.zeros(source.shape, dtype=np.uint8)
    else:
        display = np.clip(np.rint((values - minimum) * (255.0 / (maximum - minimum))), 0, 255).astype(
            np.uint8
        )
    return display


def thermal_source_display_bgr(image: np.ndarray) -> np.ndarray:
    """Return a display-only BGR preview while preserving the source dtype and channels."""
    validate_image_array(image, name="thermal image")
    source = image.copy()
    source_intensity, _ = _source_to_intensity(source)
    normalized, _, _ = normalize_thermal_intensity(source_intensity)
    return _source_display_bgr(source, normalized)


def _component_candidates(
    mask: np.ndarray,
    *,
    polarity: ThermalPolarity,
    roi: ROI | None,
    config: ThermalPipelineConfig,
) -> list[ThermalComponentSelection]:
    """Rank connected components using only documented geometric criteria."""
    height, width = mask.shape
    image_area = float(height * width)
    roi_area = float(roi.width * roi.height) if roi else 1.0
    roi_mask = np.zeros(mask.shape, dtype=bool)
    if roi:
        roi_mask[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width] = True
    labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    max_center_distance = float(np.hypot(max(height - 1, 1), max(width - 1, 1)) / 2.0)
    provisional: list[tuple[int, int, int, float, float, float, float, float, float, bool, float, float]] = []
    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < config.minimum_component_area:
            continue
        component = labels == label
        overlap = int(np.count_nonzero(component & roi_mask)) if roi else 0
        relative_area = area / image_area
        # Favor a nonempty, not-frame-filling component. This is a transparent size prior,
        # not a learned person-shape model.
        area_score = min(relative_area / config.target_relative_area, 1.0) * (1.0 - relative_area)
        centroid_x, centroid_y = (float(value) for value in centroids[label])
        distance = float(np.hypot(centroid_x - (width - 1) / 2.0, centroid_y - (height - 1) / 2.0))
        centrality_score = max(0.0, 1.0 - distance / max_center_distance)
        touches_border = bool(
            np.any(component[0, :])
            or np.any(component[-1, :])
            or np.any(component[:, 0])
            or np.any(component[:, -1])
        )
        border_score = 0.0 if touches_border else 1.0
        roi_score = overlap / roi_area if roi else 0.0
        if roi:
            score = 0.60 * roi_score + 0.20 * area_score + 0.10 * centrality_score + 0.10 * border_score
        else:
            score = 0.50 * area_score + 0.35 * centrality_score + 0.15 * border_score
        provisional.append(
            (
                label,
                area,
                overlap,
                relative_area,
                roi_score,
                area_score,
                centrality_score,
                border_score,
                score,
                touches_border,
                centroid_x,
                centroid_y,
            )
        )
    provisional.sort(key=lambda item: (-item[8], -item[2], -item[1], item[0]))
    candidate_count = len(provisional)
    candidates: list[ThermalComponentSelection] = []
    for (
        label,
        area,
        overlap,
        relative_area,
        roi_score,
        area_score,
        centrality_score,
        border_score,
        score,
        touches_border,
        centroid_x,
        centroid_y,
    ) in provisional:
        candidates.append(
            ThermalComponentSelection(
                polarity=polarity,
                label=label,
                area=area,
                relative_area=relative_area,
                roi_overlap=overlap,
                roi_score=roi_score,
                area_score=area_score,
                centrality_score=centrality_score,
                border_score=border_score,
                score=score,
                centroid_x=centroid_x,
                centroid_y=centroid_y,
                touches_border=touches_border,
                candidate_count=candidate_count,
                mask=(labels == label).copy(),
            )
        )
    return candidates


def _best_component(
    mask: np.ndarray,
    *,
    polarity: ThermalPolarity,
    roi: ROI | None,
    config: ThermalPipelineConfig,
) -> ThermalComponentSelection | None:
    candidates = _component_candidates(mask, polarity=polarity, roi=roi, config=config)
    return candidates[0] if candidates else None


def run_thermal_segmentation(
    image: np.ndarray,
    roi: ROI | Sequence[int] | None = None,
    *,
    config: ThermalPipelineConfig = ThermalPipelineConfig(),
) -> ThermalSegmentationResult:
    """Run the classical thermal pipeline and return all meaningful intermediate stages."""
    validate_image_array(image, name="thermal image")
    config = _validate_config(config)
    original = image.copy()
    source_intensity, input_representation = _source_to_intensity(original)
    validated_roi = validate_roi(roi, original.shape) if roi is not None else None
    normalized, intensity_min, intensity_max = normalize_thermal_intensity(source_intensity)
    enhanced = normalized.copy()
    if config.gaussian_blur_kernel_size > 1:
        enhanced = cv2.GaussianBlur(
            enhanced,
            (config.gaussian_blur_kernel_size, config.gaussian_blur_kernel_size),
            sigmaX=0,
        )

    bright_threshold, bright_uint8 = cv2.threshold(
        enhanced, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    dark_threshold, dark_uint8 = cv2.threshold(
        enhanced, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    bright_raw = bright_uint8 > 0
    dark_raw = dark_uint8 > 0
    bright_cleaned = apply_morphological_cleanup(
        bright_raw,
        opening_kernel_size=config.opening_kernel_size,
        closing_kernel_size=config.closing_kernel_size,
    )
    dark_cleaned = apply_morphological_cleanup(
        dark_raw,
        opening_kernel_size=config.opening_kernel_size,
        closing_kernel_size=config.closing_kernel_size,
    )
    bright_component = _best_component(
        bright_cleaned, polarity="bright", roi=validated_roi, config=config
    )
    dark_component = _best_component(
        dark_cleaned, polarity="dark", roi=validated_roi, config=config
    )
    warnings: list[str] = []
    if input_representation.startswith("three-channel false-color"):
        warnings.append(
            "Three-channel input was converted from a false-color BGR palette for intensity "
            "processing; values are not calibrated temperature."
        )
    polarity_ambiguous = False
    selected_component: ThermalComponentSelection | None = None
    selected_polarity: ThermalPolarity | None = None
    if intensity_min == intensity_max:
        status = "constant"
        warnings.append("The thermal intensity image is constant; no foreground was selected.")
    else:
        viable = [
            component
            for component in (bright_component, dark_component)
            if component is not None and component.score >= config.minimum_selection_score
        ]
        if not viable:
            if bright_component is not None and dark_component is not None:
                status = "ambiguous"
                polarity_ambiguous = True
                warnings.append(
                    "Both bright and dark polarity candidates were weak; no polarity was "
                    "selected as a human foreground."
                )
            else:
                status = "empty"
                warnings.append("Neither bright nor dark polarity produced a sufficiently strong component.")
        else:
            viable.sort(key=lambda component: (-component.score, -component.roi_overlap, -component.area, component.label))
            selected_component = viable[0]
            selected_polarity = selected_component.polarity
            if len(viable) == 2 and abs(viable[0].score - viable[1].score) <= config.polarity_ambiguity_margin:
                polarity_ambiguous = True
                status = "ambiguous"
                warnings.append(
                    "Bright and dark polarity scores are near-equal; the higher-scoring candidate "
                    "was selected deterministically, but polarity is ambiguous."
                )
            else:
                status = "selected"
            if selected_component.touches_border:
                warnings.append("The selected component touches the image border; interpret the result cautiously.")
    final_mask = (
        selected_component.mask.copy()
        if selected_component is not None and status != "constant"
        else np.zeros(normalized.shape, dtype=bool)
    )
    contours = extract_external_contours(final_mask)
    source_display_bgr = _source_display_bgr(original, normalized)
    boundary_overlay_bgr = draw_contours_on_bgr(
        source_display_bgr,
        contours,
        thickness=config.contour_thickness,
    )
    parameters: dict[str, int | float] = {
        "gaussian_blur_kernel_size": int(config.gaussian_blur_kernel_size),
        "opening_kernel_size": int(config.opening_kernel_size),
        "closing_kernel_size": int(config.closing_kernel_size),
        "minimum_component_area": int(config.minimum_component_area),
        "target_relative_area": float(config.target_relative_area),
        "minimum_selection_score": float(config.minimum_selection_score),
        "polarity_ambiguity_margin": float(config.polarity_ambiguity_margin),
        "contour_thickness": int(config.contour_thickness),
    }
    return ThermalSegmentationResult(
        original_thermal=original,
        source_metadata=describe_image(original, color_order="THERMAL"),
        input_representation=input_representation,
        source_intensity=source_intensity.copy(),
        intensity_min=intensity_min,
        intensity_max=intensity_max,
        normalized_intensity=normalized,
        enhanced_intensity=enhanced,
        bright_otsu_threshold=float(bright_threshold),
        dark_otsu_threshold=float(dark_threshold),
        bright_raw_mask=bright_raw,
        dark_raw_mask=dark_raw,
        bright_cleaned_mask=bright_cleaned,
        dark_cleaned_mask=dark_cleaned,
        selected_polarity=selected_polarity,
        polarity_ambiguous=polarity_ambiguous,
        bright_component=bright_component,
        dark_component=dark_component,
        selected_component=selected_component,
        final_mask=final_mask,
        contours=contours,
        source_display_bgr=source_display_bgr,
        boundary_overlay_bgr=boundary_overlay_bgr,
        status=status,
        parameters=parameters,
        warnings=tuple(warnings),
    )
