"""SAM2 comparison: recorded-evidence lookup, dual-boundary rendering, and per-page numbers.

The recorded SAM2 masks are real official outputs committed with the repository. These tests
check that they are found only for pixel-identical frames, and that every number the RGB and
thermal pages show is recomputed from the live classical mask and those recorded masks.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = str(REPO_ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from module4.io_utils import load_image_bgr, load_image_unchanged  # noqa: E402
from module4.metrics import evaluate_masks  # noqa: E402
from module4.reference import find_recorded_sam2_case, load_sam2_records, load_timing_records  # noqa: E402
from module4.rgb import run_rgb_segmentation  # noqa: E402
from module4.thermal import run_thermal_segmentation  # noqa: E402
from module4.types import RGBPipelineConfig, ThermalPipelineConfig  # noqa: E402
from module4.validation import canonicalize_mask  # noqa: E402
from module4.visualization import (  # noqa: E402
    CLASSICAL_BOUNDARY_BGR,
    SAM2_BOUNDARY_BGR,
    boundary_comparison_bgr,
)
from module4.webapp import pages  # noqa: E402


def _on_fresh_thread(function):
    out = {}
    worker = threading.Thread(target=lambda: out.setdefault("value", function()))
    worker.start()
    worker.join()
    return out["value"]


def test_six_completed_sam2_records_with_real_masks() -> None:
    records = load_sam2_records(REPO_ROOT)
    assert {(r["image_id"].rsplit("-", 1)[-1], r["modality"]) for r in records} == {
        (frame, modality) for frame in ("00085", "00135", "00185") for modality in ("rgb", "thermal")
    }
    for record in records:
        assert record["status"] == "completed"
        assert record["prompt"]["derived_from_classical_mask"] is False
        mask = canonicalize_mask(load_image_unchanged(REPO_ROOT / record["reference_image"]), name="sam2")
        assert int(mask.sum()) == record["sam2"]["foreground_pixels"]


@pytest.mark.parametrize("modality", ["rgb", "thermal"])
def test_bundled_frame_matches_its_recorded_case(modality: str) -> None:
    path = pages._SAMPLE_RGB_IMAGE if modality == "rgb" else pages._SAMPLE_THERMAL_IMAGE
    image = load_image_bgr(path) if modality == "rgb" else load_image_unchanged(path)
    case = find_recorded_sam2_case(image, modality, REPO_ROOT)
    assert case is not None
    assert case.experiment_id == f"aau-vap-scene1-00085-{modality}"
    assert (case.prompt.x, case.prompt.y, case.prompt.width, case.prompt.height) == pages._SAMPLE_ROI.as_tuple()
    assert case.ground_truth_path is not None and case.ground_truth_path.is_file()


def test_lookup_is_by_content_not_name_or_modality() -> None:
    image = load_image_bgr(pages._SAMPLE_RGB_IMAGE)
    changed = image.copy()
    changed[0, 0, 0] ^= 1  # one pixel differs: not the recorded frame
    assert find_recorded_sam2_case(changed, "rgb", REPO_ROOT) is None
    assert find_recorded_sam2_case(np.zeros((480, 640, 3), np.uint8), "rgb", REPO_ROOT) is None
    with pytest.raises(ValueError):
        find_recorded_sam2_case(image, "depth", REPO_ROOT)


def test_missing_evidence_files_mean_no_records(tmp_path: Path) -> None:
    assert load_sam2_records(tmp_path) == []
    assert load_timing_records(tmp_path) is None
    assert find_recorded_sam2_case(load_image_bgr(pages._SAMPLE_RGB_IMAGE), "rgb", tmp_path) is None


def test_timing_record_reproduced_the_frozen_evidence() -> None:
    timing = load_timing_records(REPO_ROOT)
    assert timing is not None and len(timing["cases"]) == 6
    for case in timing["cases"]:
        assert case["classical_matches_frozen_mask"] is True
        for device_result in case["sam2"].values():
            assert device_result["rerun_equals_recorded"] is True
            assert device_result["total_median_ms"] > 0


def test_boundary_comparison_draws_both_colors_without_mutating_input() -> None:
    image = np.full((40, 60, 3), 90, np.uint8)
    original = image.copy()
    classical = np.zeros((40, 60), bool)
    classical[5:35, 5:30] = True
    sam2 = np.zeros((40, 60), bool)
    sam2[10:30, 35:55] = True
    overlay = boundary_comparison_bgr(image, classical, sam2)
    assert np.array_equal(image, original)
    colors = {tuple(pixel) for pixel in overlay.reshape(-1, 3)}
    assert CLASSICAL_BOUNDARY_BGR in colors and SAM2_BOUNDARY_BGR in colors
    with pytest.raises(ValueError):
        boundary_comparison_bgr(image, classical[:20], sam2[:20])


RGB_SCRIPT = f"""
import sys
sys.path.insert(0, {SRC!r})
from module4.webapp import pages
pages._rgb_page()
"""

THERMAL_SCRIPT = f"""
import sys
sys.path.insert(0, {SRC!r})
from module4.webapp import pages
pages._thermal_page()
"""


@pytest.mark.parametrize("modality", ["rgb", "thermal"])
def test_page_sam2_section_numbers_are_recomputed_live(modality: str) -> None:
    if modality == "rgb":
        image = load_image_bgr(pages._SAMPLE_RGB_IMAGE)
        classical = _on_fresh_thread(lambda: run_rgb_segmentation(
            image, pages._SAMPLE_ROI, config=RGBPipelineConfig(**pages._SAMPLE_RGB_PARAMS)).final_mask)
        script = RGB_SCRIPT
    else:
        image = load_image_unchanged(pages._SAMPLE_THERMAL_IMAGE)
        classical = _on_fresh_thread(lambda: run_thermal_segmentation(
            image, pages._SAMPLE_ROI, config=ThermalPipelineConfig(**pages._SAMPLE_THERMAL_PARAMS)).final_mask)
        script = THERMAL_SCRIPT
    case = find_recorded_sam2_case(image, modality, REPO_ROOT)
    truth = canonicalize_mask(load_image_unchanged(case.ground_truth_path), name="truth")

    app = AppTest.from_string(script, default_timeout=180).run()
    assert not app.exception
    assert "Comparison with SAM2" in [s.value for s in app.subheader]
    vs_sam2 = evaluate_masks(classical, case.mask)
    assert [(m.label, m.value) for m in app.metric] == [
        ("IoU", f"{vs_sam2.iou:.4f}"),
        ("Dice", f"{vs_sam2.dice:.4f}"),
        ("Precision", f"{vs_sam2.precision:.4f}"),
        ("Recall", f"{vs_sam2.recall:.4f}"),
    ]
    (table,) = app.dataframe
    assert list(table.value["IoU"]) == [
        round(evaluate_masks(classical, truth).iou, 4),
        round(evaluate_masks(case.mask, truth).iou, 4),
        round(vs_sam2.iou, 4),
    ]
    assert any(f"{vs_sam2.fp} pixels" in m.value and f"{vs_sam2.fn} pixels" in m.value for m in app.markdown)
