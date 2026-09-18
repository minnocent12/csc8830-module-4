from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from module4.rgb import run_rgb_segmentation, validate_roi
from module4.types import ROI, RGBPipelineConfig


def test_validate_roi_accepts_valid_tuple_and_rejects_invalid_values() -> None:
    shape = (20, 30, 3)
    assert validate_roi((2, 3, 10, 8), shape) == ROI(2, 3, 10, 8)

    invalid_rois = [
        (-1, 2, 10, 8),
        (2, -1, 10, 8),
        (2, 3, 0, 8),
        (2, 3, 10, 0),
        (25, 3, 10, 8),
        (2, 15, 10, 8),
        (2, 3, 29, 18),
    ]
    for roi in invalid_rois:
        with pytest.raises(ValueError):
            validate_roi(roi, shape)

    with pytest.raises(TypeError):
        validate_roi((1, 2, 3), shape)
    with pytest.raises(TypeError):
        validate_roi(("1", 2, 3, 4), shape)  # type: ignore[arg-type]


def test_validate_roi_rejects_partial_out_of_bounds_instead_of_clamping() -> None:
    with pytest.raises(ValueError, match="fully inside"):
        validate_roi((25, 3, 10, 8), (20, 30, 3))


def _synthetic_color_scene() -> tuple[np.ndarray, np.ndarray, ROI]:
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    image[:] = (40, 130, 40)
    expected = np.zeros((80, 100), dtype=bool)
    expected[18:68, 38:63] = True
    image[18:68, 38:63] = (30, 30, 220)
    return image, expected, ROI(30, 10, 40, 65)


def test_rgb_pipeline_returns_meaningful_classical_result_without_mutating_input() -> None:
    image, expected, roi = _synthetic_color_scene()
    original = image.copy()

    result = run_rgb_segmentation(
        image,
        roi,
        config=RGBPipelineConfig(
            grabcut_iterations=5,
            opening_kernel_size=1,
            closing_kernel_size=1,
            minimum_component_area=10,
        ),
    )

    assert result.final_mask.dtype == bool
    assert result.final_mask.shape == image.shape[:2]
    assert result.boundary_overlay_bgr.shape == image.shape
    assert result.grabcut_labels.shape == image.shape[:2]
    assert result.raw_foreground_mask.shape == image.shape[:2]
    assert result.cleaned_foreground_mask.shape == image.shape[:2]
    assert result.roi == roi
    assert result.component_selection is not None
    assert np.count_nonzero(result.final_mask) > 0
    overlap = np.count_nonzero(result.final_mask & expected)
    assert overlap > 500
    np.testing.assert_array_equal(image, original)


def test_rgb_pipeline_preserves_separate_color_diagnostics() -> None:
    image, _, roi = _synthetic_color_scene()
    result = run_rgb_segmentation(image, roi)

    assert result.grayscale.shape == image.shape[:2]
    assert result.hsv.shape == image.shape
    assert result.lab.shape == image.shape
    assert result.hsv.dtype == np.uint8
    assert result.lab.dtype == np.uint8


def test_rgb_pipeline_rejects_non_bgr_uint8_input() -> None:
    with pytest.raises(ValueError, match="color channels"):
        run_rgb_segmentation(np.zeros((10, 10), dtype=np.uint8), ROI(1, 1, 5, 5))
    with pytest.raises(ValueError, match="uint8 BGR"):
        run_rgb_segmentation(np.zeros((10, 10, 3), dtype=np.uint16), ROI(1, 1, 5, 5))


def test_rgb_pipeline_rejects_invalid_parameters() -> None:
    image, _, roi = _synthetic_color_scene()
    with pytest.raises(ValueError, match="positive"):
        run_rgb_segmentation(image, roi, config=RGBPipelineConfig(grabcut_iterations=0))
    with pytest.raises(ValueError, match="odd"):
        run_rgb_segmentation(image, roi, config=RGBPipelineConfig(opening_kernel_size=2))
    with pytest.raises(TypeError, match="integer"):
        run_rgb_segmentation(image, roi, config=RGBPipelineConfig(contour_thickness=1.5))  # type: ignore[arg-type]


def test_component_failure_is_structured_for_empty_selection() -> None:
    from module4.components import select_roi_component

    assert select_roi_component(np.zeros((8, 8), dtype=bool), ROI(1, 1, 4, 4)) is None


def test_phase2_source_contains_no_prohibited_ml_dependencies() -> None:
    prohibited = {
        "torch",
        "tensorflow",
        "mediapipe",
        "transformers",
        "ultralytics",
        "detectron",
        "sam2",
    }
    source_root = Path(__file__).parents[1] / "src" / "module4"
    import_lines = []
    for path in (source_root / "rgb.py", source_root / "preprocessing.py", source_root / "components.py", source_root / "contours.py"):
        for line in path.read_text(encoding="utf-8").lower().splitlines():
            if line.strip().startswith(("import ", "from ")):
                import_lines.append(line)
    assert not any(package in line for line in import_lines for package in prohibited)
