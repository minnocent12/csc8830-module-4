"""Comparison and Evaluation page: structure, reference provenance, and exact metric display.

These run on the bundled AAU VAP frame (no AppTest uploads), so they work on every supported
Streamlit version. The expected numbers are recomputed here from the same classical pipeline
and reference mask the page uses, never typed in.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = str(REPO_ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from module4.io_utils import load_image_bgr, load_image_unchanged  # noqa: E402
from module4.metrics import evaluate_masks  # noqa: E402
from module4.rgb import run_rgb_segmentation  # noqa: E402
from module4.types import RGBPipelineConfig  # noqa: E402
from module4.validation import prepare_reference  # noqa: E402
from module4.webapp import pages  # noqa: E402

PAGE_SCRIPT = f"""
import sys
sys.path.insert(0, {SRC!r})
from module4.webapp import pages
pages._comparison_page()
"""

# Presentation-only double: a "completed" SAM2 result built from the bundled ground-truth
# mask, used solely to exercise the page's success layout. It is not a SAM2 output.
STUB_SCRIPT = f"""
import sys
sys.path.insert(0, {SRC!r})
import cv2
from module4.types import SAM2ReferenceResult
from module4.webapp import pages

def _stub(image_rgb, prompt, *, config, source_image_id=None):
    mask = cv2.imread({str(pages._SAMPLE_REFERENCE_RGB_MASK)!r}, cv2.IMREAD_UNCHANGED)
    return SAM2ReferenceResult(
        status="completed", mask=mask, model_name=config.model_name,
        model_config=config.model_config, checkpoint_identifier="test-double",
        checkpoint_source=None, implementation_source="test-double", implementation_version=None,
        device="cpu", prompt_type="box", prompt=prompt, source_image_id=source_image_id,
        source_dimensions=image_rgb.shape[:2], selected_mask_index=0, predictor_scores=(1.0,),
        selection_rule="test double", provenance={{}}, warnings=(), error=None)

# The page module is shared with every other test in this process, so restore it afterwards.
original = pages.run_sam2_reference
pages.run_sam2_reference = _stub
try:
    pages._comparison_page()
finally:
    pages.run_sam2_reference = original
"""


def _html(app: AppTest) -> str:
    return " ".join(e.proto.body for e in app.get("html") if "<style" not in e.proto.body)


def _expected_bundled_rgb_metrics():
    """Recompute on a fresh thread: OpenCV's RNG (used by GrabCut) is thread-local, and each
    Streamlit script run starts on a new thread with the default RNG state."""
    out = {}
    worker = threading.Thread(target=lambda: out.setdefault("metrics", _compute_bundled_rgb_metrics()))
    worker.start()
    worker.join()
    return out["metrics"]


def _compute_bundled_rgb_metrics():
    source = load_image_bgr(pages._SAMPLE_RGB_IMAGE)
    result = run_rgb_segmentation(
        source, pages._SAMPLE_ROI, config=RGBPipelineConfig(**pages._SAMPLE_RGB_PARAMS)
    )
    prepared = prepare_reference(
        load_image_unchanged(pages._SAMPLE_REFERENCE_RGB_MASK),
        expected_shape=result.final_mask.shape,
        reference_status="available",
        reference_type="ground_truth",
        image_id=pages._SAMPLE_RGB_IMAGE.name,
        reference_image_id=pages._SAMPLE_RGB_IMAGE.name,
    )
    return evaluate_masks(result.final_mask, prepared.mask)


@pytest.fixture(scope="module")
def bundled() -> AppTest:
    app = AppTest.from_string(PAGE_SCRIPT, default_timeout=180).run()
    assert not app.exception
    return app


def test_header_is_canonical_and_shown_once(bundled: AppTest) -> None:
    assert [h.value for h in bundled.header] == ["Supporting evaluation: Comparison and Evaluation"]
    assert "Module 4" in _html(bundled)


def test_sections_follow_the_evaluation_workflow(bundled: AppTest) -> None:
    assert [s.value for s in bundled.subheader] == [
        "Input",
        "Reference",
        "Classical Prediction",
        "Evaluation",
        "Interpretation",
    ]
    cards = [m.value for m in bundled.markdown if m.value.startswith("#### ")]
    assert cards == ["#### Classical Configuration", "#### Reference Configuration"]


def test_metrics_and_confusion_counts_match_the_pipeline_exactly(bundled: AppTest) -> None:
    expected = _expected_bundled_rgb_metrics()
    assert [(m.label, m.value) for m in bundled.metric] == [
        ("IoU", f"{expected.iou:.4f}"),
        ("Dice", f"{expected.dice:.4f}"),
        ("Precision", f"{expected.precision:.4f}"),
        ("Recall", f"{expected.recall:.4f}"),
    ]
    (table,) = bundled.dataframe
    rows = dict(zip(table.value["Count"], table.value["Pixels"]))
    assert rows == {"TP": expected.tp, "FP": expected.fp, "FN": expected.fn, "TN": expected.tn}


def test_reference_provenance_is_always_stated(bundled: AppTest) -> None:
    html = _html(bundled)
    assert "Reference: ground truth" in html
    assert "Source: bundled AAU VAP ground-truth mask" in html
    assert "Alignment: none" in html
    assert any("bundled real ground-truth mask" in item.value for item in bundled.info)


def test_no_reference_is_an_explanation_not_a_warning() -> None:
    app = AppTest.from_string(PAGE_SCRIPT, default_timeout=180).run()
    app.radio(key="comparison_reference_source").set_value("None").run()
    assert not app.exception
    assert not app.metric
    assert not app.warning
    assert any("No reference selected" in item.value for item in app.info)


def test_sam2_unavailable_stays_a_warning_and_offers_the_uploaded_alternative() -> None:
    app = AppTest.from_string(PAGE_SCRIPT, default_timeout=180).run()
    app.radio(key="comparison_reference_source").set_value("SAM2 reference").run()
    assert not app.exception
    warnings = [item.value for item in app.warning]
    assert any("SAM2 reference is unavailable" in w and "Uploaded reference mask" in w for w in warnings)
    assert not app.metric
    assert not app.dataframe
    assert "Evaluation" not in [s.value for s in app.subheader]
    assert "SAM2 Reference" in [s.value for s in app.subheader]


def test_completed_sam2_reference_is_labeled_as_sam2_not_ground_truth() -> None:
    app = AppTest.from_string(STUB_SCRIPT, default_timeout=180).run()
    app.radio(key="comparison_reference_source").set_value("SAM2 reference").run()
    app.checkbox(key="comparison_sam2_reuse_roi").check().run()
    assert not app.exception
    html = _html(app)
    assert "Reference: SAM2 reference segmentation" in html
    assert "Source: SAM2 run on this image" in html
    assert "ground truth" not in html
    assert {m.label for m in app.metric} == {"IoU", "Dice", "Precision", "Recall"}
    assert any("no ground-truth claim is made" in m.value for m in app.markdown)
