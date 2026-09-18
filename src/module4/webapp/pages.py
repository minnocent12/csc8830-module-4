"""Module 4 Streamlit pages and the get_pages provider."""
from __future__ import annotations

import streamlit as st

from module4.io_utils import decode_image_bgr, decode_image_unchanged
from module4.metrics import evaluate_masks
from module4.rgb import run_rgb_segmentation
from module4.thermal import run_thermal_segmentation, thermal_source_display_bgr
from module4.types import ROI, RGBPipelineConfig, ThermalPipelineConfig
from module4.validation import (
    ReferenceValidationError,
    align_reference_mask,
    prepare_reference,
)
from module4.webapp._page import PageSpec
from module4.webapp.ui import IMAGE_TYPES, foundation_page, pending_experiment_banner
from module4.visualization import bgr_to_rgb, display_mask, mask_overlap_bgr, to_display_uint8

_MODULE = "Module 4"


def _rgb_page() -> None:
    st.header("RGB Human Boundary")
    st.info(
        "This page uses ROI-assisted classical OpenCV processing. You provide the rectangle "
        "around the person; no machine learning or deep learning is used."
    )
    upload = st.file_uploader("RGB image", type=IMAGE_TYPES)
    if upload is None:
        pending_experiment_banner("Upload an RGB image to run the Phase 2 pipeline.")
        return
    try:
        image_bgr = decode_image_bgr(upload.getvalue(), source_name=upload.name)
    except (TypeError, ValueError) as exc:
        st.error(f"Could not read the RGB image: {exc}")
        return

    height, width = image_bgr.shape[:2]
    st.caption(f"Input: {upload.name} | {width} x {height} pixels | OpenCV BGR uint8")
    st.image(bgr_to_rgb(image_bgr), caption="Original RGB image", width="stretch")
    if width < 2 or height < 2:
        st.error("The image must be at least 2 x 2 pixels for ROI-assisted GrabCut.")
        return

    st.subheader("User-provided ROI")
    st.caption("The ROI is strict xywh: x and y are the upper-left pixel; width and height are pixels.")
    c1, c2, c3, c4 = st.columns(4)
    x = int(c1.number_input("x", min_value=0, max_value=width - 2, value=width // 4, step=1))
    y = int(c2.number_input("y", min_value=0, max_value=height - 2, value=height // 8, step=1))
    roi_width = int(
        c3.number_input(
            "width",
            min_value=2,
            max_value=width - x,
            value=min(max(2, width // 2), width - x),
            step=1,
        )
    )
    roi_height = int(
        c4.number_input(
            "height",
            min_value=2,
            max_value=height - y,
            value=min(max(2, (height * 3) // 4), height - y),
            step=1,
        )
    )
    iterations = int(
        st.slider("GrabCut iterations", min_value=1, max_value=15, value=5, help="More iterations can refine the classical optimization.")
    )
    opening_size = int(st.select_slider("Opening kernel", options=[1, 3, 5, 7], value=3))
    closing_size = int(st.select_slider("Closing kernel", options=[1, 3, 5, 7], value=5))
    st.caption(f"Selected ROI: x={x}, y={y}, width={roi_width}, height={roi_height}")

    if not st.button("Run classical RGB segmentation", type="primary"):
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

    st.subheader("Classical OpenCV intermediate results")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(bgr_to_rgb(result.roi_overlay_bgr), caption="Original with user ROI", width="stretch")
    with c2:
        st.image(to_display_uint8(result.grayscale), caption="Grayscale diagnostic", width="stretch")
    with c3:
        st.image(display_mask(result.raw_foreground_mask), caption="Raw GrabCut foreground", width="stretch")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(display_mask(result.cleaned_foreground_mask), caption="After morphology", width="stretch")
    with c2:
        st.image(display_mask(result.final_mask), caption="Selected final human mask", width="stretch")
    with c3:
        st.image(bgr_to_rgb(result.boundary_overlay_bgr), caption="Final contour on original", width="stretch")

    if result.component_selection is None:
        st.warning("No valid foreground component was selected. The final mask is empty.")
    else:
        st.write(
            f"Selected component label {result.component_selection.label}; "
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
    st.header("Thermal Human Boundary")
    st.info(
        "This implementation uses classical OpenCV image processing only. Human/background "
        "polarity is not assumed in advance; bright and dark Otsu candidates are both evaluated."
    )
    upload = st.file_uploader("Thermal or thermal-intensity image", type=IMAGE_TYPES)
    if upload is None:
        pending_experiment_banner("Upload a thermal image to run the Phase 3 pipeline.")
        return
    try:
        image = decode_image_unchanged(upload.getvalue(), source_name=upload.name)
    except (TypeError, ValueError) as exc:
        st.error(f"Could not read the thermal image: {exc}")
        return

    height, width = image.shape[:2]
    st.caption(
        f"Input: {upload.name} | {width} x {height} pixels | source dtype: {image.dtype} | "
        "three-channel inputs are treated as false-color BGR palettes"
    )
    try:
        source_preview_bgr = thermal_source_display_bgr(image)
    except (TypeError, ValueError) as exc:
        st.error(f"Thermal source is not supported: {exc}")
        return
    st.image(
        bgr_to_rgb(source_preview_bgr),
        caption="Original source (display-only scaling; computational source is preserved)",
        width="stretch",
    )

    use_roi = st.checkbox("Use an optional ROI to strengthen component selection", value=False)
    roi = None
    if use_roi:
        if width < 2 or height < 2:
            st.error("An ROI requires an image at least 2 x 2 pixels; disable ROI to continue.")
            return
        st.caption("The ROI is strict xywh: x and y are the upper-left pixel; width and height are pixels.")
        c1, c2, c3, c4 = st.columns(4)
        x = int(c1.number_input("x", min_value=0, max_value=width - 2, value=width // 4, step=1))
        y = int(c2.number_input("y", min_value=0, max_value=height - 2, value=height // 4, step=1))
        roi_width = int(
            c3.number_input(
                "width",
                min_value=2,
                max_value=width - x,
                value=min(max(2, width // 2), width - x),
                step=1,
            )
        )
        roi_height = int(
            c4.number_input(
                "height",
                min_value=2,
                max_value=height - y,
                value=min(max(2, height // 2), height - y),
                step=1,
            )
        )
        roi = ROI(x, y, roi_width, roi_height)
        st.caption(f"Selected ROI: x={x}, y={y}, width={roi_width}, height={roi_height}")

    gaussian_size = int(st.select_slider("Gaussian denoising kernel", options=[1, 3, 5], value=1))
    opening_size = int(st.select_slider("Opening kernel", options=[1, 3, 5, 7], value=3))
    closing_size = int(st.select_slider("Closing kernel", options=[1, 3, 5, 7], value=5))
    if not st.button("Run classical thermal segmentation", type="primary"):
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
        st.image(bgr_to_rgb(result.source_display_bgr), caption="Source for display", width="stretch")
    with c2:
        st.image(result.normalized_intensity, caption="Normalized intensity", width="stretch")
    with c3:
        st.image(result.enhanced_intensity, caption="Enhanced intensity", width="stretch")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.image(display_mask(result.bright_raw_mask), caption=f"Bright Otsu raw (t={result.bright_otsu_threshold:.2f})", width="stretch")
    with c2:
        st.image(display_mask(result.dark_raw_mask), caption=f"Dark Otsu raw (t={result.dark_otsu_threshold:.2f})", width="stretch")
    with c3:
        st.image(display_mask(result.bright_cleaned_mask), caption="Bright after morphology", width="stretch")
    with c4:
        st.image(display_mask(result.dark_cleaned_mask), caption="Dark after morphology", width="stretch")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(display_mask(result.final_mask), caption="Selected final mask", width="stretch")
    with c2:
        st.image(bgr_to_rgb(result.boundary_overlay_bgr), caption="Final boundary overlay", width="stretch")
    with c3:
        st.write(f"Status: **{result.status}**")
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
        "metrics. SAM2, Fourier theory, and empirical RGB-versus-thermal conclusions remain "
        "pending later approved phases and real data."
    )


def _comparison_page() -> None:
    st.header("Comparison and Evaluation")
    st.info(
        "Run one of the independent classical pipelines first, then provide a reference mask. "
        "The reference is used only after segmentation and never changes the predicted mask."
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
        st.subheader("RGB pipeline parameters")
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
        st.image(
            bgr_to_rgb(source_preview),
            caption="Thermal source preview (display scaling only)",
            width="stretch",
        )
        st.subheader("Thermal pipeline parameters")
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

    st.subheader("Reference mask")
    reference_upload = st.file_uploader(
        "Optional binary reference mask",
        type=IMAGE_TYPES,
        key="comparison_reference",
        help="Accepted mask pixels are bool-equivalent uint8 values 0, 1, and 255. Other grayscale values are rejected.",
    )
    reference_type = st.selectbox(
        "Reference type",
        ["ground_truth", "user_reference", "sam2_reference"],
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
    if reference_type == "sam2_reference":
        st.caption("This label records user-supplied provenance only. No SAM2 model is run in Phase 4.")

    if not st.button("Run classical pipeline and evaluate", type="primary"):
        pending_experiment_banner("Run the classical pipeline to produce a prediction and evaluate it if a reference is supplied.")
        return

    try:
        if modality == "rgb":
            result = run_rgb_segmentation(source, roi, config=config)
        else:
            result = run_thermal_segmentation(source, roi, config=config)
    except (TypeError, ValueError) as exc:
        st.error(f"Classical {modality_label.lower()} segmentation could not run: {exc}")
        return

    st.subheader("Classical prediction")
    st.image(display_mask(result.final_mask), caption="Predicted foreground mask", width="stretch")
    st.image(bgr_to_rgb(result.boundary_overlay_bgr), caption="Predicted boundary overlay", width="stretch")
    if modality == "thermal":
        st.write(f"Selected polarity: **{result.selected_polarity or 'none'}**; status: **{result.status}**")
    for warning in result.warnings:
        st.warning(warning)

    if reference_upload is None:
        st.warning("Reference mask is pending; IoU, Dice, precision, recall, and confusion counts are unavailable.")
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
        st.error(f"Reference validation failed; metrics were not generated: {exc}")
        return

    st.subheader("Validated comparison")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(display_mask(reference_mask), caption="Validated reference mask", width="stretch")
    with c2:
        st.image(bgr_to_rgb(mask_overlap_bgr(result.final_mask, reference_mask)), caption="Overlap: TP green, FP red, FN blue", width="stretch")
    with c3:
        st.write(f"Reference type: **{reference_type.replace('_', ' ')}**")
        st.write(f"Alignment: **{alignment.transformation if alignment else 'none'}**")
        st.write(f"Reference dimensions: **{reference_mask.shape[1]} x {reference_mask.shape[0]}**")
    st.subheader("Pixel-level metrics")
    metric_columns = st.columns(4)
    for column, label, value in zip(
        metric_columns,
        ("IoU", "Dice", "Precision", "Recall"),
        (metrics.iou, metrics.dice, metrics.precision, metrics.recall),
    ):
        column.metric(label, f"{value:.4f}")
    st.write(
        f"TP: **{metrics.tp}** | FP: **{metrics.fp}** | FN: **{metrics.fn}** | TN: **{metrics.tn}**"
    )
    st.caption(
        "All metrics use aligned canonical masks. A reference is metadata/provenance input, not a "
        "segmentation input; no SAM2 inference or Fourier processing is performed here."
    )


def _theory_page() -> None:
    foundation_page(
        "Fourier Theory",
        "The Parts A-F Fourier-domain derivation and optional demonstrations will be added in a "
        "later approved phase.",
    )


def get_pages() -> list[PageSpec]:
    """Return the Module 4 pages for standalone or shared-dashboard hosts."""
    return [
        PageSpec(_MODULE, "RGB Human Boundary", 10, _rgb_page),
        PageSpec(_MODULE, "Thermal Human Boundary", 20, _thermal_page),
        PageSpec(_MODULE, "Comparison and Evaluation", 30, _comparison_page),
        PageSpec(_MODULE, "Fourier Theory", 40, _theory_page),
    ]
