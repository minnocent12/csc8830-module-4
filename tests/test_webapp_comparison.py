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

# Simulates an image that is not a recorded dataset frame, so live SAM2 is the only source.
NOT_RECORDED_SCRIPT = f"""
import sys
sys.path.insert(0, {SRC!r})
from module4.webapp import pages

original = pages.find_recorded_sam2_case
pages.find_recorded_sam2_case = lambda *args, **kwargs: None
try:
    pages._comparison_page()
finally:
    pages.find_recorded_sam2_case = original
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
        "SAM2 Comparison Results",
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
    # Recorded SAM2 results, measured timing, then this run's confusion counts.
    assert len(bundled.dataframe) == 3
    table = bundled.dataframe[-1]
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
    app = AppTest.from_string(NOT_RECORDED_SCRIPT, default_timeout=180).run()
    app.radio(key="comparison_reference_source").set_value("SAM2 reference").run()
    assert not app.exception
    warnings = [item.value for item in app.warning]
    assert any("SAM2 reference is unavailable" in w and "Uploaded reference mask" in w for w in warnings)
    assert not app.metric
    assert len(app.dataframe) == 2  # only the recorded results and timing tables
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


def test_recorded_sam2_results_table_matches_the_records() -> None:
    import json

    app = AppTest.from_string(PAGE_SCRIPT, default_timeout=180).run()
    records = json.loads((REPO_ROOT / "results/metrics/phase8_sam2_experiment_records.json").read_text())
    table = app.dataframe[0].value
    assert len(table) == 3 * len(records) == 18
    expected = []
    for record in sorted(records, key=lambda row: (row["image_id"], row["modality"])):
        for scores in (
            record["dataset_ground_truth"]["metrics_for_frozen_classical"],
            record["comparisons"]["sam2_vs_dataset_ground_truth"],
            record["comparisons"]["classical_vs_sam2"],
        ):
            expected.append(round(scores["iou"], 4))
    assert list(table["IoU"]) == expected
    assert list(table["Method"][:3]) == ["Classical OpenCV", "SAM2", "Classical OpenCV"]
    assert list(table["Compared with"][:3]) == ["ground truth", "ground truth", "SAM2"]


def test_measured_timing_table_matches_the_timing_record() -> None:
    import json

    app = AppTest.from_string(PAGE_SCRIPT, default_timeout=180).run()
    timing = json.loads((REPO_ROOT / "results/metrics/phase8_timing_records.json").read_text())
    table = app.dataframe[1].value
    assert list(table["Classical OpenCV, CPU (ms)"]) == [
        round(case["classical"]["median_ms"], 1) for case in timing["cases"]
    ]
    for device in timing["sam2_model_load_ms"]:
        assert list(table[f"SAM2, {device.upper()} (ms)"]) == [
            round(case["sam2"][device]["total_median_ms"], 1) for case in timing["cases"]
        ]


def test_bundled_frame_uses_the_recorded_official_sam2_mask_when_sam2_cannot_run() -> None:
    from module4.reference import find_recorded_sam2_case

    app = AppTest.from_string(PAGE_SCRIPT, default_timeout=180).run()
    app.radio(key="comparison_reference_source").set_value("SAM2 reference").run()
    assert not app.exception
    assert not app.warning
    html = _html(app)
    assert "Reference: SAM2 reference segmentation" in html
    assert "Source: recorded official SAM2 run on this frame" in html
    recorded = find_recorded_sam2_case(load_image_bgr(pages._SAMPLE_RGB_IMAGE), "rgb", REPO_ROOT)
    out = {}
    worker = threading.Thread(target=lambda: out.setdefault("mask", run_rgb_segmentation(
        load_image_bgr(pages._SAMPLE_RGB_IMAGE), pages._SAMPLE_ROI,
        config=RGBPipelineConfig(**pages._SAMPLE_RGB_PARAMS)).final_mask))
    worker.start()
    worker.join()
    expected = evaluate_masks(out["mask"], recorded.mask)
    assert [(m.label, m.value) for m in app.metric] == [
        ("IoU", f"{expected.iou:.4f}"),
        ("Dice", f"{expected.dice:.4f}"),
        ("Precision", f"{expected.precision:.4f}"),
        ("Recall", f"{expected.recall:.4f}"),
    ]
