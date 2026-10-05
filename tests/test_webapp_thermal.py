"""Thermal Human Boundary page: structure, unchanged dual-polarity results, and status semantics.

The pinned values were recorded from the classical thermal pipeline before the page was
reorganized; they fix the algorithm contract (selected polarity, Otsu thresholds, selected
component, final mask and overlay). The page tests then check that what the page shows is
exactly what the pipeline computes. No test uploads a file, so all run on Streamlit 1.49.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

SRC = str(Path(__file__).resolve().parents[1] / "src")
PAGE_SCRIPT = f"""
import sys
sys.path.insert(0, {SRC!r})
from module4.webapp import pages
pages._thermal_page()
"""

import sys  # noqa: E402

if SRC not in sys.path:
    sys.path.insert(0, SRC)

from module4.io_utils import load_image_unchanged  # noqa: E402
from module4.thermal import run_thermal_segmentation  # noqa: E402
from module4.types import ThermalPipelineConfig  # noqa: E402
from module4.webapp import pages  # noqa: E402


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


def _bundled(roi):
    return run_thermal_segmentation(
        load_image_unchanged(pages._SAMPLE_THERMAL_IMAGE),
        roi,
        config=ThermalPipelineConfig(**pages._SAMPLE_THERMAL_PARAMS),
    )


def _body(polarity: str) -> np.ndarray:
    image = np.full((120, 160), 40, np.uint8)
    image[25:105, 60:100] = 210
    return image if polarity == "hot" else 255 - image


@pytest.mark.parametrize(
    ("case", "polarity", "threshold", "area", "border", "final_mask", "overlay"),
    [
        ("bundled with predefined ROI", "dark", 98.0, 214632, True, "76967cec3224b768", "721c06654e9b6c86"),
        ("bundled without ROI", "bright", 98.0, 49278, True, "2045c34b43dda331", "77736e13de762ea0"),
    ],
)
def test_bundled_frame_dual_polarity_contract(case, polarity, threshold, area, border, final_mask, overlay) -> None:
    result = _bundled(pages._SAMPLE_ROI if "predefined" in case else None)
    assert result.selected_polarity == polarity
    assert result.bright_otsu_threshold == threshold
    assert result.dark_otsu_threshold == threshold
    component = result.selected_component
    assert (component.label, component.area, component.touches_border) == (1, area, border)
    assert _digest(result.final_mask) == final_mask
    assert _digest(result.boundary_overlay_bgr) == overlay


@pytest.mark.parametrize(("body", "polarity"), [("hot", "bright"), ("cold", "dark")])
def test_polarity_follows_the_foreground_contrast(body: str, polarity: str) -> None:
    result = run_thermal_segmentation(_body(body), None, config=ThermalPipelineConfig())
    assert result.selected_polarity == polarity
    assert result.selected_component.area == 3196
    assert _digest(result.final_mask) == "c799c3bbe706eb7f"


@pytest.fixture(scope="module")
def page() -> AppTest:
    app = AppTest.from_string(PAGE_SCRIPT, default_timeout=180).run()
    assert not app.exception
    return app


def test_header_is_canonical_and_shown_once(page: AppTest) -> None:
    assert [h.value for h in page.header] == ["Question 2: Thermal Human Boundary"]
    assert "Module 4" in " ".join(e.proto.body for e in page.get("html"))


def test_sections_follow_the_workflow(page: AppTest) -> None:
    assert [s.value for s in page.subheader] == ["Input", "Processing Results", "Interpretation"]
    assert "#### Configuration" in [m.value for m in page.markdown]


def test_page_shows_exactly_what_the_pipeline_computes(page: AppTest) -> None:
    result = _bundled(pages._SAMPLE_ROI)
    text = [m.value for m in page.markdown]
    assert f"Selected polarity: **{result.selected_polarity}**" in text
    component = result.selected_component
    assert (
        f"Component label {component.label}; area {component.area} pixels; "
        f"score {component.score:.3f}; border contact: {component.touches_border}."
    ) in text
    captions = [img.proto.imgs[0].caption for img in page.get("image") or page.get("imgs")]
    assert captions == [
        "Source thermal/intensity display (display-only scaling; computational source is preserved)",
        "Source thermal/intensity display",
        "Normalized thermal intensity",
        "Gaussian-enhanced thermal intensity",
        f"Bright Otsu candidate (t={result.bright_otsu_threshold:.2f})",
        f"Dark Otsu candidate (t={result.dark_otsu_threshold:.2f})",
        "Bright candidate after morphology",
        "Dark candidate after morphology",
        "Final classical thermal mask",
        "Boundary overlay derived from final thermal mask",
    ]


def test_pipeline_summary_is_ordinary_text_not_a_success_badge(page: AppTest) -> None:
    assert not page.success
    assert not any("Thermal pipeline" in item.value for item in page.info)
    assert any(
        m.value == "Both bright and dark foreground hypotheses were evaluated before selecting the dark classical component."
        for m in page.markdown
    )
    # Real limitations stay warnings.
    assert any("touches the image border" in item.value for item in page.warning)


def test_controls_keep_labels_and_bundled_defaults(page: AppTest) -> None:
    assert page.checkbox[0].label == "Use an optional ROI to strengthen component selection"
    assert page.checkbox[0].value is True
    assert {w.label: w.value for w in page.number_input} == {"x": 120, "y": 20, "width": 450, "height": 460}
    assert [(w.label, w.value) for w in page.select_slider] == [
        ("Gaussian denoising kernel", 1),
        ("Opening kernel", 3),
        ("Closing kernel", 5),
    ]
    assert not page.button  # the bundled frame runs immediately; uploads keep the run button


def test_turning_the_roi_off_selects_the_bright_hypothesis_on_the_page() -> None:
    app = AppTest.from_string(PAGE_SCRIPT, default_timeout=180).run()
    app.checkbox[0].uncheck().run()
    assert not app.exception
    assert "Selected polarity: **bright**" in [m.value for m in app.markdown]
