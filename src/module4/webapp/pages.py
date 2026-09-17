"""Module 4 Streamlit pages and the get_pages provider."""
from __future__ import annotations

import streamlit as st

from module4.io_utils import decode_image_bgr
from module4.rgb import run_rgb_segmentation
from module4.types import ROI, RGBPipelineConfig
from module4.webapp._page import PageSpec
from module4.webapp.ui import IMAGE_TYPES, foundation_page, pending_experiment_banner
from module4.visualization import bgr_to_rgb, display_mask, to_display_uint8

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
        "SAM2 reference comparison and IoU/Dice/precision/recall are intentionally not present "
        "until their later approved phases."
    )


def _thermal_page() -> None:
    foundation_page(
        "Thermal Human Boundary",
        "Thermal image loading, normalization, classical polarity handling, contour extraction, "
        "and reference comparison will be added in later phases.",
    )


def _comparison_page() -> None:
    foundation_page(
        "Comparison and Evaluation",
        "Mask alignment, IoU, Dice, precision, recall, visual comparisons, and result downloads "
        "will be added after the independent pipelines exist.",
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
