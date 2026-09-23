"""Module 4 Streamlit pages and the get_pages provider."""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from module4.fourier import (
    apply_frequency_filter,
    compute_fft2,
    frequency_derivative,
    frequency_laplacian,
    gaussian_high_pass,
    gaussian_low_pass,
    local_frequency_energy,
    magnitude_spectrum,
)
from module4.io_utils import (
    decode_image_bgr,
    decode_image_unchanged,
    load_image_bgr,
    load_image_unchanged,
)
from module4.metrics import evaluate_masks
from module4.reference import (
    bgr_to_sam2_rgb,
    run_sam2_reference,
    thermal_to_sam2_rgb,
)
from module4.rgb import run_rgb_segmentation
from module4.thermal import run_thermal_segmentation, thermal_source_display_bgr
from module4.types import (
    ROI,
    RGBPipelineConfig,
    SAM2Config,
    SAM2Prompt,
    SegmentationMetrics,
    ThermalPipelineConfig,
)
from module4.validation import (
    ReferenceValidationError,
    align_reference_mask,
    prepare_reference,
)
from module4.webapp._page import PageSpec
from module4.webapp.ui import (
    IMAGE_TYPES,
    bundled_sample_notice,
    page_header,
    pending_experiment_banner,
    status_message,
)
from module4.visualization import bgr_to_rgb, display_mask, mask_overlap_bgr, to_display_uint8

_MODULE = "Module 4"
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Real Phase 8 case: aau-vap-scene1-frame-00085 (AAU VAP Trimodal People Segmentation
# Dataset, CC BY 4.0). ROI and pipeline parameters copied from
# data/experiment_manifest.json, where they were predefined before any reference comparison.
_SAMPLE_RGB_IMAGE = _REPO_ROOT / "data" / "rgb" / "aau_vap_scene1_00085.jpg"
_SAMPLE_THERMAL_IMAGE = _REPO_ROOT / "data" / "thermal" / "aau_vap_scene1_00085.jpg"
_SAMPLE_ROI = ROI(120, 20, 450, 460)
_SAMPLE_RGB_PARAMS = {"grabcut_iterations": 5, "opening_kernel_size": 3, "closing_kernel_size": 5}
_SAMPLE_THERMAL_PARAMS = {"gaussian_blur_kernel_size": 1, "opening_kernel_size": 3, "closing_kernel_size": 5}


def _rgb_page() -> None:
    page_header(
        "RGB Human Boundary",
        assignment_label="Question 1",
        summary=(
            "Demonstrate ROI-assisted classical OpenCV human-boundary segmentation. You supply "
            "the ROI; the final contour is derived from the final classical mask."
        ),
        input_hint="RGB color image (OpenCV BGR uint8).",
    )
    upload = st.file_uploader("RGB image", type=IMAGE_TYPES)

    using_sample = False
    if upload is not None:
        try:
            image_bgr = decode_image_bgr(upload.getvalue(), source_name=upload.name)
        except (TypeError, ValueError) as exc:
            st.error(f"Could not read the RGB image: {exc}")
            return
        source_name = upload.name
    elif _SAMPLE_RGB_IMAGE.is_file():
        using_sample = True
        image_bgr = load_image_bgr(_SAMPLE_RGB_IMAGE)
        source_name = _SAMPLE_RGB_IMAGE.name
        bundled_sample_notice(
            "No upload — showing a live demo on a bundled real frame from the AAU VAP "
            "Trimodal People Segmentation Dataset (CC BY 4.0), with the ROI pre-filled at "
            "the real Phase 8 predefined location. Upload your own RGB image to override."
        )
    else:
        pending_experiment_banner("Upload an RGB image to run the Phase 2 pipeline.")
        return

    height, width = image_bgr.shape[:2]
    st.caption(f"Input: {source_name} | {width} x {height} pixels | OpenCV BGR uint8")
    st.image(bgr_to_rgb(image_bgr), caption="Original RGB image", width="stretch")
    if width < 2 or height < 2:
        st.error("The image must be at least 2 x 2 pixels for ROI-assisted GrabCut.")
        return

    st.subheader("User-supplied ROI")
    st.caption("Classical RGB processing is not automatic: the rectangle must be supplied around the person.")
    st.caption("The ROI is strict xywh: x and y are the upper-left pixel; width and height are pixels.")
    default_x = _SAMPLE_ROI.x if using_sample else width // 4
    default_y = _SAMPLE_ROI.y if using_sample else height // 8
    default_w = _SAMPLE_ROI.width if using_sample else min(max(2, width // 2), width)
    default_h = _SAMPLE_ROI.height if using_sample else min(max(2, (height * 3) // 4), height)
    c1, c2, c3, c4 = st.columns(4)
    x = int(c1.number_input("x", min_value=0, max_value=width - 2, value=min(default_x, width - 2), step=1))
    y = int(c2.number_input("y", min_value=0, max_value=height - 2, value=min(default_y, height - 2), step=1))
    roi_width = int(
        c3.number_input(
            "width",
            min_value=2,
            max_value=width - x,
            value=min(max(2, default_w), width - x),
            step=1,
        )
    )
    roi_height = int(
        c4.number_input(
            "height",
            min_value=2,
            max_value=height - y,
            value=min(max(2, default_h), height - y),
            step=1,
        )
    )
    iterations = int(
        st.slider(
            "GrabCut iterations",
            min_value=1,
            max_value=15,
            value=_SAMPLE_RGB_PARAMS["grabcut_iterations"] if using_sample else 5,
            help="More iterations can refine the classical optimization.",
        )
    )
    opening_size = int(
        st.select_slider(
            "Opening kernel",
            options=[1, 3, 5, 7],
            value=_SAMPLE_RGB_PARAMS["opening_kernel_size"] if using_sample else 3,
        )
    )
    closing_size = int(
        st.select_slider(
            "Closing kernel",
            options=[1, 3, 5, 7],
            value=_SAMPLE_RGB_PARAMS["closing_kernel_size"] if using_sample else 5,
        )
    )
    st.caption(f"Selected ROI: x={x}, y={y}, width={roi_width}, height={roi_height}")

    if not using_sample and not st.button("Run classical RGB segmentation", type="primary"):
        pending_experiment_banner("Set the ROI and run the classical pipeline to view intermediate results.")
        return

    try:
        result = run_rgb_segmentation(
            image_bgr,
            ROI(x, y, roi_width, roi_height),
            config=RGBPipelineConfig(
                grabcut_iterations=iterations,
                opening_kernel_size=opening_size,
                closing_kernel_size=closing_size,
            ),
        )
    except (TypeError, ValueError) as exc:
        st.error(f"RGB segmentation could not run: {exc}")
        return

    st.subheader("Processing sequence")
    st.caption(
        "Original → user ROI → GrabCut → morphology → selected component → final classical mask → "
        "contour/boundary overlay"
    )
    st.subheader("Classical OpenCV intermediate results")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(bgr_to_rgb(result.roi_overlay_bgr), caption="Original RGB image with user ROI", width="stretch")
    with c2:
        st.image(to_display_uint8(result.grayscale), caption="Grayscale diagnostic", width="stretch")
    with c3:
        st.image(display_mask(result.raw_foreground_mask), caption="GrabCut raw foreground candidate", width="stretch")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(display_mask(result.cleaned_foreground_mask), caption="Morphology-cleaned foreground candidate", width="stretch")
    with c2:
        st.image(display_mask(result.final_mask), caption="Final classical RGB mask", width="stretch")
    with c3:
        st.image(bgr_to_rgb(result.boundary_overlay_bgr), caption="Boundary overlay derived from final mask", width="stretch")

    if result.component_selection is None:
        st.warning("No valid foreground component was selected. The final mask is empty.")
    else:
        st.write(
            f"Selected component from final classical mask: label {result.component_selection.label}; "
            f"area {result.component_selection.area} pixels; "
            f"ROI overlap {result.component_selection.roi_overlap} pixels."
        )
    for warning in result.warnings:
        st.warning(warning)
    st.caption(
        "Use Comparison and Evaluation to upload a reference and compute validated IoU, Dice, "
        "precision, recall, and confusion counts."
    )


def _thermal_page() -> None:
    page_header(
        "Thermal Human Boundary",
        assignment_label="Question 2",
        summary=(
            "Evaluate bright and dark thermal foreground hypotheses with classical OpenCV "
            "processing before selecting a cleaned component and boundary."
        ),
        input_hint=(
            "Single-channel thermal/intensity image or supported false-color palette; palette "
            "colors are not calibrated temperature."
        ),
    )
    upload = st.file_uploader("Thermal or thermal-intensity image", type=IMAGE_TYPES)

    using_sample = False
    if upload is not None:
        try:
            image = decode_image_unchanged(upload.getvalue(), source_name=upload.name)
        except (TypeError, ValueError) as exc:
            st.error(f"Could not read the thermal image: {exc}")
            return
        source_name = upload.name
    elif _SAMPLE_THERMAL_IMAGE.is_file():
        using_sample = True
        image = load_image_unchanged(_SAMPLE_THERMAL_IMAGE)
        source_name = _SAMPLE_THERMAL_IMAGE.name
        bundled_sample_notice(
            "No upload — showing a live demo on a bundled real thermal frame from the AAU "
            "VAP Trimodal People Segmentation Dataset (CC BY 4.0), with the ROI pre-filled "
            "at the real Phase 8 predefined location. Upload your own thermal image to "
            "override."
        )
    else:
        pending_experiment_banner("Upload a thermal image to run the Phase 3 pipeline.")
        return

    height, width = image.shape[:2]
    st.caption(
        f"Input: {source_name} | {width} x {height} pixels | source dtype: {image.dtype} | "
        "three-channel inputs are treated as false-color BGR palettes"
    )
    try:
        source_preview_bgr = thermal_source_display_bgr(image)
    except (TypeError, ValueError) as exc:
        st.error(f"Thermal source is not supported: {exc}")
        return
    st.subheader("Source and optional ROI")
    if image.ndim == 3 and image.shape[2] == 3:
        st.warning("This input is a false-color BGR palette. Palette colors are not calibrated physical temperature.")
    st.caption(
        "Both bright and dark foreground hypotheses are evaluated; the selected polarity is a "
        "transparent classical choice, not an assumption that the person is hotter."
    )
    st.image(
        bgr_to_rgb(source_preview_bgr),
        caption="Source thermal/intensity display (display-only scaling; computational source is preserved)",
        width="stretch",
    )

    use_roi = st.checkbox("Use an optional ROI to strengthen component selection", value=using_sample)
    roi = None
    if use_roi:
        if width < 2 or height < 2:
            st.error("An ROI requires an image at least 2 x 2 pixels; disable ROI to continue.")
            return
        st.caption("The ROI is strict xywh: x and y are the upper-left pixel; width and height are pixels.")
        default_x = _SAMPLE_ROI.x if using_sample else width // 4
        default_y = _SAMPLE_ROI.y if using_sample else height // 4
        default_w = _SAMPLE_ROI.width if using_sample else min(max(2, width // 2), width)
        default_h = _SAMPLE_ROI.height if using_sample else min(max(2, height // 2), height)
        c1, c2, c3, c4 = st.columns(4)
        x = int(c1.number_input("x", min_value=0, max_value=width - 2, value=min(default_x, width - 2), step=1))
        y = int(c2.number_input("y", min_value=0, max_value=height - 2, value=min(default_y, height - 2), step=1))
        roi_width = int(
            c3.number_input(
                "width",
                min_value=2,
                max_value=width - x,
                value=min(max(2, default_w), width - x),
                step=1,
            )
        )
        roi_height = int(
            c4.number_input(
                "height",
                min_value=2,
                max_value=height - y,
                value=min(max(2, default_h), height - y),
                step=1,
            )
        )
        roi = ROI(x, y, roi_width, roi_height)
        st.caption(f"Selected ROI: x={x}, y={y}, width={roi_width}, height={roi_height}")

    gaussian_size = int(
        st.select_slider(
            "Gaussian denoising kernel",
            options=[1, 3, 5],
            value=_SAMPLE_THERMAL_PARAMS["gaussian_blur_kernel_size"] if using_sample else 1,
        )
    )
    opening_size = int(
        st.select_slider(
            "Opening kernel",
            options=[1, 3, 5, 7],
            value=_SAMPLE_THERMAL_PARAMS["opening_kernel_size"] if using_sample else 3,
        )
    )
    closing_size = int(
        st.select_slider(
            "Closing kernel",
            options=[1, 3, 5, 7],
            value=_SAMPLE_THERMAL_PARAMS["closing_kernel_size"] if using_sample else 5,
        )
    )
    if not using_sample and not st.button("Run classical thermal segmentation", type="primary"):
        pending_experiment_banner("Configure optional preprocessing/ROI settings and run the thermal pipeline.")
        return

    try:
        result = run_thermal_segmentation(
            image,
            roi,
            config=ThermalPipelineConfig(
                gaussian_blur_kernel_size=gaussian_size,
                opening_kernel_size=opening_size,
                closing_kernel_size=closing_size,
            ),
        )
    except (TypeError, ValueError) as exc:
        st.error(f"Thermal segmentation could not run: {exc}")
        return

    st.subheader("Classical thermal intermediate results")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(bgr_to_rgb(result.source_display_bgr), caption="Source thermal/intensity display", width="stretch")
    with c2:
        st.image(result.normalized_intensity, caption="Normalized thermal intensity", width="stretch")
    with c3:
        st.image(result.enhanced_intensity, caption="Gaussian-enhanced thermal intensity", width="stretch")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.image(display_mask(result.bright_raw_mask), caption=f"Bright Otsu candidate (t={result.bright_otsu_threshold:.2f})", width="stretch")
    with c2:
        st.image(display_mask(result.dark_raw_mask), caption=f"Dark Otsu candidate (t={result.dark_otsu_threshold:.2f})", width="stretch")
    with c3:
        st.image(display_mask(result.bright_cleaned_mask), caption="Bright candidate after morphology", width="stretch")
    with c4:
        st.image(display_mask(result.dark_cleaned_mask), caption="Dark candidate after morphology", width="stretch")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(display_mask(result.final_mask), caption="Final classical thermal mask", width="stretch")
    with c2:
        st.image(bgr_to_rgb(result.boundary_overlay_bgr), caption="Boundary overlay derived from final thermal mask", width="stretch")
    with c3:
        status_message(
            "Thermal pipeline",
            "Implemented",
            "Both bright and dark foreground hypotheses were evaluated before selecting the "
            f"{result.selected_polarity or 'available'} classical component.",
        )
        st.write(f"Selected polarity: **{result.selected_polarity or 'none'}**")
        if result.selected_component is not None:
            component = result.selected_component
            st.write(
                f"Component label {component.label}; area {component.area} pixels; "
                f"score {component.score:.3f}; border contact: {component.touches_border}."
            )
    for warning in result.warnings:
        st.warning(warning)
    st.caption(
        "Use Comparison and Evaluation to upload a reference and compute validated pixel-level "
        "metrics. The tracked Phase 8 records contain six fixed real-data classical and official "
        "SAM2 reference comparisons; the optional SAM2 runtime still requires its isolated "
        "environment and local checkpoint. Fourier theory is available on the Fourier Theory page."
    )


def _sam2_prompt_controls(width: int, height: int, classical_roi: ROI | None) -> SAM2Prompt | None:
    """Collect an independent SAM2 box, optionally reusing the user's classical ROI."""
    reuse_roi = st.checkbox(
        "Reuse the same user-supplied rectangle for SAM2",
        value=False,
        key="comparison_sam2_reuse_roi",
        help="This reuses the rectangle you supplied above; it is not derived from a classical mask or component.",
    )
    if reuse_roi:
        if classical_roi is None:
            st.warning("No classical ROI was supplied. Enter an independent SAM2 box instead.")
        else:
            st.caption("SAM2 and the classical pipeline will receive the same user-supplied rectangle.")
            return SAM2Prompt(*classical_roi.as_tuple())

    default_x = min(width - 1, max(0, width // 4))
    default_y = min(height - 1, max(0, height // 4))
    c1, c2, c3, c4 = st.columns(4)
    prompt_x = int(c1.number_input("SAM2 prompt x", min_value=0, max_value=width - 1, value=default_x, step=1))
    prompt_y = int(c2.number_input("SAM2 prompt y", min_value=0, max_value=height - 1, value=default_y, step=1))
    prompt_width = int(
        c3.number_input(
            "SAM2 prompt width",
            min_value=1,
            max_value=width - prompt_x,
            value=min(max(1, width // 2), width - prompt_x),
            step=1,
        )
    )
    prompt_height = int(
        c4.number_input(
            "SAM2 prompt height",
            min_value=1,
            max_value=height - prompt_y,
            value=min(max(1, height // 2), height - prompt_y),
            step=1,
        )
    )
    return SAM2Prompt(prompt_x, prompt_y, prompt_width, prompt_height)


def _sam2_configuration() -> SAM2Config:
    """Collect optional SAM2 settings without making them base-application requirements."""
    with st.expander("Optional SAM2 runtime configuration"):
        model_name = st.selectbox(
            "SAM2 model",
            ["sam2.1_hiera_tiny", "sam2.1_hiera_small", "sam2.1_hiera_base_plus", "sam2.1_hiera_large"],
            key="comparison_sam2_model",
        )
        config_names = {
            "sam2.1_hiera_tiny": "sam2.1_hiera_t.yaml",
            "sam2.1_hiera_small": "sam2.1_hiera_s.yaml",
            "sam2.1_hiera_base_plus": "sam2.1_hiera_b+.yaml",
            "sam2.1_hiera_large": "sam2.1_hiera_l.yaml",
        }
        default_config = f"configs/sam2.1/{config_names[model_name]}"
        model_config = st.text_input("SAM2 model config", value=default_config, key="comparison_sam2_config")
        checkpoint_path = st.text_input(
            "Local SAM2 checkpoint path",
            value=os.environ.get("MODULE4_SAM2_CHECKPOINT", ""),
            key="comparison_sam2_checkpoint",
            help="The base app never downloads a checkpoint. Use a local path from the separate official SAM2 environment.",
        ).strip() or None
        device = st.selectbox("SAM2 device", ["auto", "cpu", "cuda", "mps"], key="comparison_sam2_device")
        implementation_version = st.text_input(
            "SAM2 implementation version (optional)",
            value="",
            key="comparison_sam2_version",
        ).strip() or None
    return SAM2Config(
        model_name=model_name,
        model_config=model_config,
        checkpoint_path=checkpoint_path,
        implementation_version=implementation_version,
        device=device,
    )


def _show_metrics(result_mask, reference_mask, *, reference_label: str, alignment: object) -> SegmentationMetrics:
    """Render the shared validated pixel-level comparison panel."""
    metrics = evaluate_masks(result_mask, reference_mask)
    st.subheader("4. Validation, overlap, and metrics")
    st.caption("Reference status: Available. Metrics use validated canonical masks.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(display_mask(reference_mask), caption=f"{reference_label} mask", width="stretch")
    with c2:
        st.image(
            bgr_to_rgb(mask_overlap_bgr(result_mask, reference_mask)),
            caption="Overlap: TP green, FP red, FN blue",
            width="stretch",
        )
    with c3:
        st.write(f"Reference type: **{reference_label}**")
        st.write("Reference status: **Available**")
        st.write(f"Alignment: **{alignment.transformation if alignment else 'none'}**")
        st.write(f"Reference dimensions: **{reference_mask.shape[1]} x {reference_mask.shape[0]}**")
    st.subheader("IoU, Dice, Precision, Recall, and confusion counts")
    metric_columns = st.columns(4)
    for column, label, value in zip(
        metric_columns,
        ("IoU", "Dice", "Precision", "Recall"),
        (metrics.iou, metrics.dice, metrics.precision, metrics.recall),
    ):
        column.metric(label, f"{value:.4f}")
    st.write(f"TP: **{metrics.tp}** | FP: **{metrics.fp}** | FN: **{metrics.fn}** | TN: **{metrics.tn}**")
    return metrics


def _fourier_display(values: np.ndarray, *, signed: bool = False) -> np.ndarray:
    """Scale a finite scalar array for display without changing Fourier computations."""
    if not isinstance(values, np.ndarray) or values.ndim != 2:
        raise ValueError("Fourier display values must be a 2D array")
    if not np.isfinite(values).all():
        raise ValueError("Fourier display values must be finite")
    values_float = values.astype(np.float64, copy=False)
    if signed:
        maximum = float(np.max(np.abs(values_float)))
        if maximum == 0:
            return np.zeros(values.shape, dtype=np.uint8)
        return np.clip((values_float / maximum + 1.0) * 127.5, 0, 255).astype(np.uint8)
    minimum = float(np.min(values_float))
    maximum = float(np.max(values_float))
    if maximum == minimum:
        return np.zeros(values.shape, dtype=np.uint8)
    return np.clip(np.rint((values_float - minimum) * 255.0 / (maximum - minimum)), 0, 255).astype(np.uint8)


def _synthetic_fourier_example() -> np.ndarray:
    """Create a deterministic smooth/striped educational scalar image for Part F."""
    rows, columns = 128, 192
    x = np.arange(columns, dtype=np.float64)
    smooth = 0.25 + 0.45 * (x[: columns // 2] / max(columns // 2 - 1, 1))
    stripes = 0.25 + 0.65 * (np.sin(2.0 * np.pi * 0.20 * x[columns // 2 :]) > 0)
    row = np.concatenate((smooth, stripes))
    return np.repeat(row[None, :], rows, axis=0)


def _comparison_page() -> None:
    page_header(
        "Comparison and Evaluation",
        assignment_label="Supporting evaluation",
        summary=(
            "Compare a completed classical mask with an uploaded reference mask or optional SAM2 "
            "reference segmentation. Validation precedes overlap visualization and metrics."
        ),
        input_hint=(
            "RGB or thermal image, plus an optional binary reference or optional SAM2 configuration."
        ),
    )
    st.caption(
        "Flow: classical mask → reference mask → validation → overlap visualization → "
        "IoU/Dice/Precision/Recall and confusion counts."
    )
    modality_label = st.radio("Modality", ["RGB", "Thermal"], horizontal=True)
    modality = modality_label.lower()
    upload = st.file_uploader("Input image", type=IMAGE_TYPES, key="comparison_input")
    if upload is None:
        pending_experiment_banner("Upload an RGB or thermal image to run a comparison.")
        return

    try:
        if modality == "rgb":
            source = decode_image_bgr(upload.getvalue(), source_name=upload.name)
        else:
            source = decode_image_unchanged(upload.getvalue(), source_name=upload.name)
    except (TypeError, ValueError) as exc:
        st.error(f"Could not read the input image: {exc}")
        return

    height, width = source.shape[:2]
    st.caption(
        f"Input: {upload.name} | {width} x {height} pixels | "
        f"computational source dtype: {source.dtype}"
    )
    if modality == "rgb":
        st.image(bgr_to_rgb(source), caption="Original RGB image", width="stretch")
        if width < 2 or height < 2:
            st.error("The RGB image must be at least 2 x 2 pixels for ROI-assisted GrabCut.")
            return
        st.subheader("1. Classical input and parameters")
        c1, c2, c3, c4 = st.columns(4)
        x = int(c1.number_input("ROI x", min_value=0, max_value=width - 2, value=width // 4, step=1))
        y = int(c2.number_input("ROI y", min_value=0, max_value=height - 2, value=height // 8, step=1))
        roi_width = int(
            c3.number_input(
                "ROI width",
                min_value=2,
                max_value=width - x,
                value=min(max(2, width // 2), width - x),
                step=1,
            )
        )
        roi_height = int(
            c4.number_input(
                "ROI height",
                min_value=2,
                max_value=height - y,
                value=min(max(2, (height * 3) // 4), height - y),
                step=1,
            )
        )
        iterations = int(st.slider("GrabCut iterations", 1, 15, 5, key="comparison_rgb_iterations"))
        opening_size = int(st.select_slider("Opening kernel", [1, 3, 5, 7], value=3, key="comparison_rgb_opening"))
        closing_size = int(st.select_slider("Closing kernel", [1, 3, 5, 7], value=5, key="comparison_rgb_closing"))
        roi = ROI(x, y, roi_width, roi_height)
        config: RGBPipelineConfig | ThermalPipelineConfig = RGBPipelineConfig(
            grabcut_iterations=iterations,
            opening_kernel_size=opening_size,
            closing_kernel_size=closing_size,
        )
    else:
        try:
            source_preview = thermal_source_display_bgr(source)
        except (TypeError, ValueError) as exc:
            st.error(f"Thermal source is not supported: {exc}")
            return
        if source.ndim == 3 and source.shape[2] == 3:
            st.warning("This input is a false-color BGR palette. Palette colors are not calibrated physical temperature.")
        st.image(
            bgr_to_rgb(source_preview),
            caption="Thermal source preview (display scaling only)",
            width="stretch",
        )
        st.subheader("1. Classical input and parameters")
        use_roi = st.checkbox("Use an optional ROI", value=False, key="comparison_thermal_use_roi")
        roi = None
        if use_roi:
            if width < 2 or height < 2:
                st.error("An ROI requires an image at least 2 x 2 pixels.")
                return
            c1, c2, c3, c4 = st.columns(4)
            x = int(c1.number_input("ROI x", min_value=0, max_value=width - 2, value=width // 4, step=1))
            y = int(c2.number_input("ROI y", min_value=0, max_value=height - 2, value=height // 4, step=1))
            roi_width = int(
                c3.number_input(
                    "ROI width",
                    min_value=2,
                    max_value=width - x,
                    value=min(max(2, width // 2), width - x),
                    step=1,
                )
            )
            roi_height = int(
                c4.number_input(
                    "ROI height",
                    min_value=2,
                    max_value=height - y,
                    value=min(max(2, height // 2), height - y),
                    step=1,
                )
            )
            roi = ROI(x, y, roi_width, roi_height)
        gaussian_size = int(st.select_slider("Gaussian kernel", [1, 3, 5], value=1, key="comparison_thermal_gaussian"))
        opening_size = int(st.select_slider("Opening kernel", [1, 3, 5, 7], value=3, key="comparison_thermal_opening"))
        closing_size = int(st.select_slider("Closing kernel", [1, 3, 5, 7], value=5, key="comparison_thermal_closing"))
        config = ThermalPipelineConfig(
            gaussian_blur_kernel_size=gaussian_size,
            opening_kernel_size=opening_size,
            closing_kernel_size=closing_size,
        )

    st.subheader("2. Reference selection")
    reference_source = st.radio(
        "Reference source",
        ["None", "Uploaded reference mask", "SAM2 reference"],
        horizontal=True,
        key="comparison_reference_source",
    )
    reference_upload = None
    reference_type = None
    reference_image_id = upload.name
    explicit_alignment = False
    sam2_prompt = None
    sam2_config = None
    if reference_source == "Uploaded reference mask":
        reference_upload = st.file_uploader(
            "Binary reference mask",
            type=IMAGE_TYPES,
            key="comparison_reference",
            help="Accepted mask pixels are bool-equivalent uint8 values 0, 1, and 255. Other grayscale values are rejected.",
        )
        reference_type = st.selectbox(
            "Reference type",
            ["ground_truth", "user_reference"],
            format_func=lambda value: value.replace("_", " ").title(),
            key="comparison_reference_type",
        )
        reference_image_id = st.text_input(
            "Reference image ID",
            value=upload.name,
            key="comparison_reference_image_id",
            help="Must match the input image ID exactly; this prevents cross-image comparisons.",
        )
        explicit_alignment = st.checkbox(
            "Explicitly align a mismatched reference with nearest-neighbor resize",
            value=False,
            key="comparison_reference_alignment",
            help="No resizing occurs unless this control is selected.",
        )
    elif reference_source == "SAM2 reference":
        status_message(
            "SAM2 integration",
            "Implemented",
            "The optional reference adapter is available without changing the classical pipeline.",
        )
        status_message(
            "Real SAM2 reference inference",
            "Available (optional)",
            "Configure the optional official environment and local checkpoint to generate a real "
            "reference segmentation.",
        )
        st.caption(
            "SAM2 is an optional reference segmentation, not ground truth. It never changes the "
            "classical prediction."
        )
        sam2_prompt = _sam2_prompt_controls(width, height, roi)
        sam2_config = _sam2_configuration()

    if not st.button("Run classical pipeline and evaluate reference", type="primary"):
        pending_experiment_banner("Run the classical pipeline to produce a prediction and evaluate the selected reference if available.")
        return

    try:
        if modality == "rgb":
            result = run_rgb_segmentation(source, roi, config=config)
        else:
            result = run_thermal_segmentation(source, roi, config=config)
    except (TypeError, ValueError) as exc:
        st.error(f"Classical {modality_label.lower()} segmentation could not run: {exc}")
        return

    st.subheader("3. Classical mask")
    st.image(display_mask(result.final_mask), caption="Final classical mask", width="stretch")
    st.image(
        bgr_to_rgb(result.boundary_overlay_bgr),
        caption="Boundary overlay derived from final classical mask",
        width="stretch",
    )
    if modality == "thermal":
        st.write(f"Selected polarity: **{result.selected_polarity or 'none'}**; status: **{result.status}**")
    for warning in result.warnings:
        st.warning(warning)

    if reference_source == "None":
        status_message(
            "Reference status",
            "Pending",
            "Metrics are unavailable until a valid reference mask exists.",
        )
        return

    if reference_source == "SAM2 reference":
        if sam2_prompt is None or sam2_config is None:
            st.error("A valid independent SAM2 box prompt is required; no reference metrics were generated.")
            return
        try:
            sam2_input = bgr_to_sam2_rgb(source) if modality == "rgb" else thermal_to_sam2_rgb(source)
            sam2_result = run_sam2_reference(
                sam2_input,
                sam2_prompt,
                config=sam2_config,
                source_image_id=upload.name,
            )
        except (TypeError, ValueError) as exc:
            status_message(
                "SAM2 reference",
                "Failed",
                f"SAM2 reference could not be prepared; metrics were not generated: {exc}",
            )
            return
        if sam2_result.status != "completed" or sam2_result.mask is None:
            if sam2_result.status == "failed":
                status_message(
                    "SAM2 reference",
                    "Failed",
                    f"SAM2 reference inference failed; metrics were not generated: {sam2_result.error}",
                )
            else:
                status_message(
                    "SAM2 reference",
                    "Unavailable",
                    f"SAM2 reference is {sam2_result.status}; metrics were not generated. "
                    f"{sam2_result.error or 'Configure the optional official SAM2 environment and checkpoint.'}",
                )
                st.caption("No placeholder mask or metrics were generated.")
            return
        st.subheader("SAM2 reference provenance")
        st.write(
            f"Model: **{sam2_result.model_name}** | config: **{sam2_result.model_config}** | "
            f"checkpoint: **{sam2_result.checkpoint_identifier or 'not recorded'}** | device: **{sam2_result.device or 'not recorded'}**"
        )
        st.write(f"Prompt: **box XYWH {sam2_result.prompt}** | selection: **{sam2_result.selection_rule}**")
        st.caption(
            "The mask was selected using predictor-native SAM2 scores only. It is a reference segmentation, "
            "not ground truth, and was canonicalized without resizing."
        )
        try:
            prepared = prepare_reference(
                sam2_result.mask,
                expected_shape=result.final_mask.shape,
                reference_status="available",
                reference_type="sam2_reference",
                image_id=upload.name,
                reference_image_id=upload.name,
            )
            if prepared.mask is None:
                raise ReferenceValidationError("SAM2 reference validation returned no mask")
        except (ReferenceValidationError, TypeError, ValueError) as exc:
            status_message(
                "SAM2 reference validation",
                "Failed",
                f"SAM2 reference validation failed; metrics were not generated: {exc}",
            )
            return
        _show_metrics(result.final_mask, prepared.mask, reference_label="SAM2 reference segmentation", alignment=prepared.alignment)
        st.caption("Metrics compare the classical mask with the SAM2 reference segmentation; no ground-truth claim is made.")
        return

    if reference_upload is None:
        status_message(
            "Reference status",
            "Pending",
            "Uploaded reference mask is pending; IoU, Dice, Precision, Recall, and confusion counts are unavailable.",
        )
        return
    try:
        reference = decode_image_unchanged(reference_upload.getvalue(), source_name=reference_upload.name)
        if explicit_alignment:
            reference_mask, alignment = align_reference_mask(
                reference,
                result.final_mask.shape,
                prediction_image_id=upload.name,
                reference_image_id=reference_image_id,
                reference_type=reference_type,
            )
        else:
            prepared = prepare_reference(
                reference,
                expected_shape=result.final_mask.shape,
                reference_status="available",
                reference_type=reference_type,
                image_id=upload.name,
                reference_image_id=reference_image_id,
            )
            if prepared.mask is None:
                raise ReferenceValidationError("reference validation returned no mask")
            reference_mask = prepared.mask
            alignment = prepared.alignment
        metrics = evaluate_masks(result.final_mask, reference_mask)
    except (ReferenceValidationError, TypeError, ValueError) as exc:
        status_message(
            "Reference validation",
            "Failed",
            f"Reference validation failed; metrics were not generated: {exc}",
        )
        return

    _show_metrics(result.final_mask, reference_mask, reference_label=reference_type.replace("_", " "), alignment=alignment)
    st.caption(
        "All metrics use aligned canonical masks. A reference is metadata/provenance input, not a "
        "segmentation input; SAM2 is optional and Fourier analysis is provided on its separate theory page."
    )


def _theory_page() -> None:
    page_header(
        "Fourier-Domain Edge Detection and Region Segmentation",
        assignment_label="Question 3",
        summary=(
            "Teach Fourier Parts A–F and provide deterministic educational demonstrations. "
            "Frequency responses are not semantic human masks or empirical results."
        ),
        input_hint=(
            "Optional scalar demonstration image; RGB uploads are explicitly converted to grayscale."
        ),
    )

    st.subheader("Theory")
    st.caption("Parts A–F: equations, frequency interpretation, filtering, derivatives, Laplacian, and local analysis.")
    st.subheader("Part A — 2D Fourier representation")
    st.markdown(
        "`f(x,y)` is the scalar spatial image; `F(u,v)` is its frequency representation. "
        "Rows are the y axis and columns are the x axis. The implementation uses NumPy's "
        "unnormalized forward FFT and normalized inverse FFT consistently."
    )
    st.latex(r"F(u,v)=\int\!\!\int f(x,y)e^{-j2\pi(ux+vy)}\,dx\,dy")
    st.latex(r"f(x,y)=\int\!\!\int F(u,v)e^{j2\pi(ux+vy)}\,du\,dv")
    st.latex(r"F[k,l]=\sum_{m=0}^{M-1}\sum_{n=0}^{N-1}f[m,n]e^{-j2\pi(km/M+ln/N)}")
    st.latex(r"f[m,n]=\frac{1}{MN}\sum_{k=0}^{M-1}\sum_{l=0}^{N-1}F[k,l]e^{j2\pi(km/M+ln/N)}")
    st.markdown(
        "Low frequencies describe slowly varying intensity and broad structure. High frequencies "
        "describe rapid changes, fine texture, sharp transitions, and noise. Magnitude is `|F|`; "
        "phase is `angle(F)`. Both matter for exact reconstruction. `fftshift` moves zero frequency "
        "to the display center; it changes layout only. The log magnitude below is display-only."
    )

    st.subheader("Demonstration")
    st.caption("Apply the theory to an uploaded scalar image or a deterministic educational example.")
    upload = st.file_uploader(
        "Optional scalar-image demonstration input",
        type=IMAGE_TYPES,
        key="fourier_input",
        help="RGB uploads are explicitly converted from OpenCV BGR to grayscale intensity.",
    )
    if upload is None:
        scalar_image = _synthetic_fourier_example()
        source_label = "Deterministic educational example"
        status_message(
            "Demonstration status",
            "Educational Demonstration",
            "Synthetic smooth/striped input is deterministic and is not an empirical result.",
        )
    else:
        try:
            source_bgr = decode_image_bgr(upload.getvalue(), source_name=upload.name)
            scalar_image = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
            source_label = f"Uploaded image: {upload.name} (BGR → grayscale intensity)"
        except (TypeError, ValueError) as exc:
            st.error(f"Could not read the Fourier demonstration image: {exc}")
            return
        status_message(
            "Demonstration status",
            "Available",
            "Uploaded input is used only for an educational Fourier response.",
        )
    st.caption(f"{source_label}. Processing representation: finite float64 scalar intensity.")
    st.subheader("Demonstration outputs")
    st.image(_fourier_display(scalar_image), caption="Scalar image used for Fourier analysis", width="stretch")

    spectrum_shifted = compute_fft2(scalar_image, shifted=True)
    st.image(
        _fourier_display(magnitude_spectrum(spectrum_shifted)),
        caption="Centered log magnitude spectrum: log(1 + |F|)",
        width="stretch",
    )

    st.subheader("Part B — Why edges are high frequency")
    st.markdown(
        "A constant region is dominated by DC/low frequency. A smooth gradient changes slowly. "
        "A sharp step changes rapidly and requires a broad spectrum, including high frequencies. "
        "High frequency is not exclusive to edges: noise, texture, detail, and compression artifacts "
        "also contribute. Therefore a high-pass response is not automatically a binary edge map."
    )
    st.latex(r"G(u,v)=H(u,v)F(u,v),\qquad g(x,y)=\mathcal{F}^{-1}\{G(u,v)\}")

    st.subheader("Part C — Gaussian high-pass filtering")
    sigma = float(
        st.slider(
            "Gaussian low-pass sigma (cycles per pixel)",
            min_value=0.01,
            max_value=0.25,
            value=0.08,
            step=0.01,
            key="fourier_sigma",
        )
    )
    low_pass = gaussian_low_pass(scalar_image.shape, sigma=sigma, shifted=True)
    high_pass = gaussian_high_pass(scalar_image.shape, sigma=sigma, shifted=True)
    low_response = apply_frequency_filter(scalar_image, low_pass, shifted=True)
    high_response = apply_frequency_filter(scalar_image, high_pass, shifted=True)
    st.latex(r"H_{HP}(u,v)=1-H_{LP}(u,v)")
    st.latex(r"G(u,v)=H_{HP}(u,v)F(u,v),\qquad g=\mathcal{F}^{-1}\{G\}")
    c1, c2 = st.columns(2)
    with c1:
        st.image(_fourier_display(low_response), caption="Gaussian low-pass reconstruction", width="stretch")
    with c2:
        st.image(_fourier_display(high_response, signed=True), caption="Gaussian high-pass response", width="stretch")
    st.caption("A Gaussian transition avoids the stronger ringing associated with an ideal hard cutoff.")

    st.subheader("Part D — Fourier derivative property")
    st.latex(r"\mathcal{F}\{\partial f/\partial x\}=j2\pi uF(u,v)")
    st.latex(r"\mathcal{F}\{\partial f/\partial y\}=j2\pi vF(u,v)")
    st.markdown(
        "The x derivative multiplies each coefficient by a factor whose magnitude grows with |u|; "
        "the y derivative does the same with |v|. This is the mathematical link between gradients, "
        "edges, and high-frequency emphasis. The signs are preserved by reconstructing the complex "
        "derivative spectrum rather than taking its magnitude."
    )
    derivative_x = frequency_derivative(scalar_image, axis="x")
    derivative_y = frequency_derivative(scalar_image, axis="y")
    c1, c2 = st.columns(2)
    with c1:
        st.image(_fourier_display(derivative_x, signed=True), caption="Fourier x derivative", width="stretch")
    with c2:
        st.image(_fourier_display(derivative_y, signed=True), caption="Fourier y derivative", width="stretch")

    st.subheader("Part E — Frequency-domain Laplacian")
    st.latex(r"\nabla^2f=\partial^2f/\partial x^2+\partial^2f/\partial y^2")
    st.latex(r"\mathcal{F}\{\nabla^2f\}=-4\pi^2(u^2+v^2)F(u,v)")
    st.markdown(
        "The radial multiplier grows quadratically with frequency, enhancing fine detail and edges. "
        "It also strongly amplifies high-frequency noise, so the Laplacian is an edge/detail operator, "
        "not a complete segmentation method."
    )
    laplacian = frequency_laplacian(scalar_image)
    st.image(_fourier_display(laplacian, signed=True), caption="Fourier-domain Laplacian response", width="stretch")

    st.subheader("Part F — Fourier-domain region segmentation")
    st.markdown(
        "Frequency-selective segmentation can transform an image or window, isolate a radial or "
        "directional band, measure selected-band energy, reconstruct a spatial response, and then "
        "threshold or clean candidate regions. Smooth regions tend to concentrate energy at lower "
        "frequencies; fine repeating textures can produce stronger high-frequency or directional energy."
    )
    st.latex(r"F_k(u,v)=\mathcal{F}\{f_k(x,y)\}")
    st.markdown(
        "A global Fourier transform reveals what frequencies occur but not directly where they occur. "
        "Local/windowed analysis divides the image into windows, computes descriptors per window, and "
        "assigns those values to a spatial map. Small windows improve localization but reduce frequency "
        "resolution; large windows do the opposite."
    )
    window_size = int(
        st.select_slider("Local analysis window size", options=[4, 8, 16, 32], value=8, key="fourier_window")
    )
    cutoff = float(
        st.slider(
            "Selected high-frequency cutoff (cycles per pixel)",
            min_value=0.01,
            max_value=0.45,
            value=0.15,
            step=0.01,
            key="fourier_cutoff",
        )
    )
    energy_map = local_frequency_energy(scalar_image, window_size=window_size, cutoff=cutoff)
    threshold_fraction = float(
        st.slider(
            "Educational display threshold as fraction of maximum energy",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key="fourier_energy_threshold",
        )
    )
    maximum_energy = float(np.max(energy_map))
    region_map = energy_map >= maximum_energy * threshold_fraction if maximum_energy > 0 else np.zeros_like(energy_map, dtype=bool)
    c1, c2 = st.columns(2)
    with c1:
        st.image(_fourier_display(energy_map), caption="Local high-frequency energy map", width="stretch")
    with c2:
        st.image(_fourier_display(region_map), caption="Thresholded educational frequency-region map", width="stretch")
    st.caption(
        "Synthetic educational demonstration when no upload is supplied; the local map is a classical "
        "frequency descriptor, not a semantic segmentation result or empirical assignment measurement."
    )
    st.markdown(
        "Advantages: interpretable frequency-band control, efficient FFT computation, and useful "
        "periodic/directional texture analysis. Limitations: global localization loss, noise sensitivity, "
        "ringing from hard cutoffs, boundary leakage, parameter dependence, and the spatial/frequency "
        "resolution trade-off of local windows. Frequency content does not directly encode semantic objects."
    )


def get_pages() -> list[PageSpec]:
    """Return the Module 4 pages for standalone or shared-dashboard hosts."""
    return [
        PageSpec(_MODULE, "RGB Human Boundary", 10, _rgb_page),
        PageSpec(_MODULE, "Thermal Human Boundary", 20, _thermal_page),
        PageSpec(_MODULE, "Comparison and Evaluation", 30, _comparison_page),
        PageSpec(_MODULE, "Fourier Theory", 40, _theory_page),
    ]
