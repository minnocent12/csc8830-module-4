"""RGB Human Boundary page: structure, unchanged GrabCut results, and the run gate.

The pinned values were recorded from the classical RGB pipeline before the page was
reorganized; they fix the algorithm contract (ROI, raw and cleaned GrabCut masks, final mask,
overlay, selected component). GrabCut draws on OpenCV's random generator, which is
thread-local, and every Streamlit script run starts on a fresh thread; recomputations here
therefore also run on a fresh thread. No test uploads a file, so all run on Streamlit 1.49.
"""
from __future__ import annotations

import hashlib
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

SRC = str(Path(__file__).resolve().parents[1] / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from module4.io_utils import load_image_bgr  # noqa: E402
from module4.rgb import run_rgb_segmentation  # noqa: E402
from module4.types import ROI, RGBPipelineConfig  # noqa: E402
from module4.webapp import pages  # noqa: E402

PAGE_SCRIPT = f"""
import sys
sys.path.insert(0, {SRC!r})
from module4.webapp import pages
pages._rgb_page()
"""


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


def _on_fresh_thread(function):
    out = {}
    worker = threading.Thread(target=lambda: out.setdefault("value", function()))
    worker.start()
    worker.join()
    return out["value"]


def _bundled(roi: ROI):
    return _on_fresh_thread(lambda: run_rgb_segmentation(
        load_image_bgr(pages._SAMPLE_RGB_IMAGE), roi, config=RGBPipelineConfig(**pages._SAMPLE_RGB_PARAMS),
    ))


def _figure() -> np.ndarray:
    image = np.full((120, 160, 3), (30, 120, 40), np.uint8)
    image[20:110, 60:100] = (40, 60, 210)
    return image


@pytest.mark.parametrize(
    ("roi", "area", "raw", "cleaned", "final", "overlay"),
    [
        (ROI(120, 20, 450, 460), 134537, "46acb92a18930abd", "394bb9062d25b1ba", "310c55eed85c76d1", "0db1f0c1e7ef9b9a"),
        (ROI(150, 20, 300, 460), 64918, "1d909b813a8f3392", "d9dcc02c054d3f8d", "2786829c68681e24", "edea8ad331d3f4bf"),
    ],
)
def test_bundled_frame_grabcut_contract(roi, area, raw, cleaned, final, overlay) -> None:
    result = _bundled(roi)
    assert result.roi == roi
    assert _digest(result.raw_foreground_mask) == raw
    assert _digest(result.cleaned_foreground_mask) == cleaned
    assert _digest(result.final_mask) == final
    assert _digest(result.boundary_overlay_bgr) == overlay
    selection = result.component_selection
    assert (selection.label, selection.area, selection.roi_overlap) == (1, area, area)
    assert len(result.contours) == 1
    assert result.warnings == ()


def test_synthetic_figure_and_empty_result_contract() -> None:
    figure = _on_fresh_thread(lambda: run_rgb_segmentation(_figure(), ROI(40, 15, 80, 90), config=RGBPipelineConfig()))
    assert figure.component_selection.area == 3396
    assert _digest(figure.final_mask) == "c3474108a63cbe70"
    uniform = _on_fresh_thread(lambda: run_rgb_segmentation(
        np.full((64, 64, 3), 128, np.uint8), ROI(16, 8, 32, 48), config=RGBPipelineConfig(),
    ))
    assert uniform.component_selection is None
    assert uniform.contours == ()
    assert uniform.warnings == ("No foreground component met the area and ROI-overlap rules.",)


@pytest.fixture(scope="module")
def page() -> AppTest:
    app = AppTest.from_string(PAGE_SCRIPT, default_timeout=180).run()
    assert not app.exception
    return app


def test_header_is_canonical_and_shown_once(page: AppTest) -> None:
    assert [h.value for h in page.header] == ["Question 1: RGB Human Boundary"]
    assert "Module 4" in " ".join(e.proto.body for e in page.get("html"))


def test_sections_follow_the_workflow(page: AppTest) -> None:
    assert [s.value for s in page.subheader] == [
        "Input", "Processing Results", "Comparison with SAM2", "Interpretation",
    ]
    markdown = [m.value for m in page.markdown]
    assert "#### Configuration" in markdown
    assert markdown.index("**Processing sequence**") < markdown.index("**Final mask and boundary**")


def test_page_shows_exactly_what_grabcut_computes(page: AppTest) -> None:
    result = _bundled(pages._SAMPLE_ROI)
    selection = result.component_selection
    assert (
        f"Selected component from final classical mask: label {selection.label}; "
        f"area {selection.area} pixels; ROI overlap {selection.roi_overlap} pixels."
    ) in [m.value for m in page.markdown]
    captions = [img.proto.imgs[0].caption for img in page.get("image") or page.get("imgs")]
    assert captions == [
        "Original RGB image",
        "Original RGB image with user ROI",
        "Grayscale diagnostic",
        "GrabCut raw foreground candidate",
        "Morphology-cleaned foreground candidate",
        "Final classical RGB mask",
        "Boundary overlay derived from final mask",
        "Classical OpenCV boundary (red)",
        "SAM2 boundary (cyan)",
        "Both boundaries: classical red, SAM2 cyan",
        "SAM2 mask",
        "Overlap: green both, red classical only, blue SAM2 only",
    ]


def test_controls_keep_labels_help_and_bundled_defaults(page: AppTest) -> None:
    assert {w.label: w.value for w in page.number_input} == {"x": 120, "y": 20, "width": 450, "height": 460}
    (iterations,) = page.slider
    assert (iterations.label, iterations.value, iterations.min, iterations.max) == ("GrabCut iterations", 5, 1, 15)
    assert iterations.proto.help == "More iterations can refine the classical optimization."
    assert [(w.label, w.value) for w in page.select_slider] == [("Opening kernel", 3), ("Closing kernel", 5)]
    assert not page.button  # bundled frame runs immediately; uploads keep the run button
    assert not page.success


def test_roi_change_on_the_bundled_frame_recomputes() -> None:
    app = AppTest.from_string(PAGE_SCRIPT, default_timeout=180).run()
    next(w for w in app.number_input if w.label == "x").set_value(150).run()
    next(w for w in app.number_input if w.label == "width").set_value(300).run()
    assert not app.exception
    assert any("area 64918 pixels" in m.value for m in app.markdown)
