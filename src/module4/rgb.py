"""ROI-assisted classical RGB human-boundary segmentation.

The pipeline uses only OpenCV operations. A user supplies the initial rectangle; no detector,
classifier, neural network, or reference model selects the ROI or changes the result.
"""
from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import cv2
import numpy as np

from module4.components import select_roi_component
from module4.contours import draw_contours_on_bgr, draw_roi_on_bgr, extract_external_contours
from module4.io_utils import validate_image_array
from module4.preprocessing import (
    apply_morphological_cleanup,
    bgr_diagnostics,
    grabcut_foreground_mask,
    validate_kernel_size,
)
from module4.types import ROI, RGBPipelineConfig, RGBSegmentationResult

MIN_ROI_SIDE = 2


def validate_roi(
    roi: ROI | Sequence[int],
    image_shape: tuple[int, ...],
    *,
    minimum_side: int = MIN_ROI_SIDE,
) -> ROI:
    """Coerce and strictly validate an image ROI.

    The ROI must be an integer xywh rectangle fully inside the image. Partial out-of-bounds
    rectangles are rejected rather than silently clamped so the user's intent is preserved.
    """
    if len(image_shape) < 2:
        raise ValueError("image_shape must contain height and width")
    height, width = int(image_shape[0]), int(image_shape[1])
    if isinstance(roi, ROI):
        values = (roi.x, roi.y, roi.width, roi.height)
    else:
        if isinstance(roi, (str, bytes)):
            raise TypeError("ROI must be an ROI or a sequence of four integers")
        try:
            values = tuple(roi)
        except TypeError as exc:
            raise TypeError("ROI must be an ROI or a sequence of four integers") from exc
    if len(values) != 4 or not all(isinstance(value, Integral) for value in values):
        raise TypeError("ROI must contain exactly four integers: x, y, width, height")
    x, y, roi_width, roi_height = (int(value) for value in values)
    if minimum_side < 2:
        raise ValueError("minimum_side must be at least 2")
    if x < 0 or y < 0:
        raise ValueError("ROI x and y must be non-negative")
    if roi_width < minimum_side or roi_height < minimum_side:
        raise ValueError(f"ROI width and height must be at least {minimum_side}")
    if x + roi_width > width or y + roi_height > height:
        raise ValueError("ROI must lie fully inside the image; partial out-of-bounds ROIs are rejected")
    return ROI(x, y, roi_width, roi_height)


def _validate_config(config: RGBPipelineConfig) -> RGBPipelineConfig:
    """Validate the small public RGB parameter set."""
    if not isinstance(config, RGBPipelineConfig):
        raise TypeError("config must be an RGBPipelineConfig")
    for name, value in (
        ("grabcut_iterations", config.grabcut_iterations),
        ("minimum_component_area", config.minimum_component_area),
        ("contour_thickness", config.contour_thickness),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    validate_kernel_size(config.opening_kernel_size, name="opening kernel size")
    validate_kernel_size(config.closing_kernel_size, name="closing kernel size")
    return config


def run_rgb_segmentation(
    image_bgr: np.ndarray,
    roi: ROI | Sequence[int],
    *,
    config: RGBPipelineConfig = RGBPipelineConfig(),
) -> RGBSegmentationResult:
    """Run the deterministic ROI-assisted classical RGB pipeline.

    Processing order is: BGR validation and diagnostics, user ROI, GrabCut rectangle
    initialization, foreground-label conversion, morphology, ROI-overlap component selection,
    external contour extraction, and boundary rendering on a copy.
    """
    validate_image_array(image_bgr, name="RGB image", color_order="BGR")
    if image_bgr.dtype != np.uint8 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("RGB processing requires an H x W x 3 uint8 BGR image")
    config = _validate_config(config)
    validated_roi = validate_roi(roi, image_bgr.shape)
    grayscale, hsv, lab = bgr_diagnostics(image_bgr)

    labels = np.full(image_bgr.shape[:2], cv2.GC_BGD, dtype=np.uint8)
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(
            image_bgr,
            labels,
            validated_roi.as_tuple(),
            background_model,
            foreground_model,
            config.grabcut_iterations,
            cv2.GC_INIT_WITH_RECT,
        )
    except cv2.error as exc:
        raise ValueError(f"OpenCV GrabCut failed for the supplied ROI: {exc}") from exc

    grabcut_labels = labels.copy()
    raw_foreground_mask = grabcut_foreground_mask(grabcut_labels)
    cleaned_foreground_mask = apply_morphological_cleanup(
        raw_foreground_mask,
        opening_kernel_size=config.opening_kernel_size,
        closing_kernel_size=config.closing_kernel_size,
    )
    selection = select_roi_component(
        cleaned_foreground_mask,
        validated_roi,
        minimum_area=config.minimum_component_area,
    )
    warnings: list[str] = []
    if selection is None:
        selected_component_mask = np.zeros_like(cleaned_foreground_mask)
        warnings.append("No foreground component met the area and ROI-overlap rules.")
    else:
        selected_component_mask = selection.mask.copy()
        if selection.ambiguous:
            warnings.append(
                "Multiple components tied under the ROI-overlap rule; the lowest component "
                "label was selected deterministically."
            )

    final_mask = selected_component_mask.copy()
    contours = extract_external_contours(final_mask)
    original_bgr = image_bgr.copy()
    roi_overlay_bgr = draw_roi_on_bgr(original_bgr, validated_roi)
    boundary_overlay_bgr = draw_contours_on_bgr(
        original_bgr,
        contours,
        thickness=config.contour_thickness,
    )
    parameters = {
        "grabcut_iterations": int(config.grabcut_iterations),
        "opening_kernel_size": int(config.opening_kernel_size),
        "closing_kernel_size": int(config.closing_kernel_size),
        "minimum_component_area": int(config.minimum_component_area),
        "contour_thickness": int(config.contour_thickness),
    }
    return RGBSegmentationResult(
        original_bgr=original_bgr,
        roi=validated_roi,
        roi_overlay_bgr=roi_overlay_bgr,
        grayscale=grayscale,
        hsv=hsv,
        lab=lab,
        grabcut_labels=grabcut_labels,
        raw_foreground_mask=raw_foreground_mask,
        cleaned_foreground_mask=cleaned_foreground_mask,
        selected_component_mask=selected_component_mask,
        final_mask=final_mask,
        contours=contours,
        boundary_overlay_bgr=boundary_overlay_bgr,
        parameters=parameters,
        warnings=tuple(warnings),
        component_selection=selection,
    )
