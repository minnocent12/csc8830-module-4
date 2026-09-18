from __future__ import annotations

import cv2
import numpy as np
import pytest

from module4.thermal import run_thermal_segmentation, thermal_source_display_bgr
from module4.types import ROI, ThermalPipelineConfig


def _bright_fixture() -> tuple[np.ndarray, np.ndarray]:
    image = np.full((80, 100), 20, dtype=np.uint8)
    expected = np.zeros(image.shape, dtype=bool)
    expected[20:60, 40:60] = True
    image[20:60, 40:60] = 220
    return image, expected


def _dark_fixture() -> tuple[np.ndarray, np.ndarray]:
    image = np.full((80, 100), 220, dtype=np.uint8)
    expected = np.zeros(image.shape, dtype=bool)
    expected[20:60, 40:60] = True
    image[20:60, 40:60] = 20
    return image, expected


def _simple_config() -> ThermalPipelineConfig:
    return ThermalPipelineConfig(
        gaussian_blur_kernel_size=1,
        opening_kernel_size=1,
        closing_kernel_size=1,
        minimum_component_area=10,
        minimum_selection_score=0.2,
    )


def test_uint8_single_channel_preserves_source_and_returns_bool_mask() -> None:
    image, _ = _bright_fixture()
    original = image.copy()
    result = run_thermal_segmentation(image, config=_simple_config())

    assert result.original_thermal.dtype == np.uint8
    assert result.normalized_intensity.dtype == np.uint8
    assert result.final_mask.dtype == bool
    assert result.final_mask.shape == image.shape
    np.testing.assert_array_equal(image, original)


def test_uint16_input_preserves_dtype_and_normalizes_for_processing() -> None:
    image = np.zeros((10, 12), dtype=np.uint16)
    image[2:8, 4:9] = 50000
    result = run_thermal_segmentation(image, config=_simple_config())

    assert result.source_metadata.dtype == "uint16"
    assert result.original_thermal.dtype == np.uint16
    assert result.normalized_intensity.dtype == np.uint8
    assert result.intensity_min == 0
    assert result.intensity_max == 50000
    assert int(result.normalized_intensity.max()) == 255


def test_source_preview_scales_uint16_without_changing_source() -> None:
    image = np.zeros((4, 5), dtype=np.uint16)
    image[1:3, 2:4] = 65535
    original = image.copy()

    preview = thermal_source_display_bgr(image)

    assert preview.shape == (4, 5, 3)
    assert preview.dtype == np.uint8
    np.testing.assert_array_equal(image, original)


def test_constant_image_is_safe_and_does_not_claim_foreground() -> None:
    result = run_thermal_segmentation(np.full((12, 12), 42, dtype=np.uint16), config=_simple_config())

    assert result.status == "constant"
    assert result.selected_polarity is None
    assert not np.any(result.final_mask)
    assert any("constant" in warning.lower() for warning in result.warnings)


def test_false_color_input_is_explicitly_converted_and_warned() -> None:
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    image[:] = (10, 20, 30)
    image[3:9, 6:11] = (200, 100, 50)
    result = run_thermal_segmentation(image, config=_simple_config())

    assert "false-color" in result.input_representation
    assert result.source_metadata.channels == 3
    assert result.source_intensity.ndim == 2
    assert result.source_display_bgr.shape == image.shape
    assert any("not calibrated temperature" in warning for warning in result.warnings)


def test_bright_fixture_selects_bright_polarity() -> None:
    image, expected = _bright_fixture()
    result = run_thermal_segmentation(image, config=_simple_config())

    assert result.selected_polarity == "bright"
    assert result.bright_component is not None
    assert result.dark_component is not None
    assert np.count_nonzero(result.final_mask & expected) > 500


def test_dark_fixture_selects_dark_polarity() -> None:
    image, expected = _dark_fixture()
    result = run_thermal_segmentation(image, config=_simple_config())

    assert result.selected_polarity == "dark"
    assert result.dark_component is not None
    assert np.count_nonzero(result.final_mask & expected) > 500


def test_dual_polarity_masks_and_otsu_thresholds_are_returned() -> None:
    image, _ = _bright_fixture()
    result = run_thermal_segmentation(image, config=_simple_config())

    assert result.bright_raw_mask.shape == image.shape
    assert result.dark_raw_mask.shape == image.shape
    assert result.bright_cleaned_mask.dtype == bool
    assert result.dark_cleaned_mask.dtype == bool
    assert isinstance(result.bright_otsu_threshold, float)
    assert isinstance(result.dark_otsu_threshold, float)


def test_morphology_removes_isolated_pixel_through_pipeline() -> None:
    image = np.full((30, 30), 20, dtype=np.uint8)
    image[1, 1] = 220
    image[10:20, 10:20] = 220
    result = run_thermal_segmentation(
        image,
        config=ThermalPipelineConfig(
            gaussian_blur_kernel_size=1,
            opening_kernel_size=3,
            closing_kernel_size=1,
            minimum_component_area=2,
            minimum_selection_score=0.1,
        ),
    )

    assert not result.bright_cleaned_mask[1, 1]
    assert result.bright_cleaned_mask[15, 15]


def test_roi_overlap_is_a_strong_deterministic_signal() -> None:
    image = np.full((60, 80), 20, dtype=np.uint8)
    image[10:25, 8:23] = 220
    image[30:52, 55:73] = 220
    result = run_thermal_segmentation(
        image,
        ROI(8, 10, 15, 15),
        config=_simple_config(),
    )

    assert result.selected_polarity == "bright"
    assert result.selected_component is not None
    assert result.selected_component.roi_overlap > 0
    assert result.selected_component.centroid_x < 30


def test_invalid_roi_is_rejected_without_clamping() -> None:
    image, _ = _bright_fixture()
    with pytest.raises(ValueError, match="fully inside"):
        run_thermal_segmentation(image, ROI(90, 10, 20, 20), config=_simple_config())


def test_ambiguity_warning_is_returned_for_near_equal_scores() -> None:
    image = np.zeros((40, 40), dtype=np.uint8)
    image[12:28, 12:28] = 120
    result = run_thermal_segmentation(
        image,
        config=ThermalPipelineConfig(
            gaussian_blur_kernel_size=1,
            opening_kernel_size=1,
            closing_kernel_size=1,
            minimum_component_area=2,
            minimum_selection_score=0.1,
            polarity_ambiguity_margin=1.0,
        ),
    )

    assert result.polarity_ambiguous
    assert result.status == "ambiguous"
    assert any("near-equal" in warning for warning in result.warnings)


def test_border_dominated_candidate_is_reported() -> None:
    image = np.full((40, 40), 20, dtype=np.uint8)
    image[:, :8] = 220
    result = run_thermal_segmentation(image, config=_simple_config())

    assert result.selected_component is not None
    assert result.selected_component.touches_border
    assert any("border" in warning.lower() for warning in result.warnings)


def test_final_contour_and_overlay_are_separate_and_non_destructive() -> None:
    image, _ = _bright_fixture()
    original = image.copy()
    result = run_thermal_segmentation(image, config=_simple_config())

    assert result.final_mask.dtype == bool
    assert result.final_mask.ndim == 2
    assert len(result.contours) >= 1
    assert result.boundary_overlay_bgr.shape == (80, 100, 3)
    assert np.any(result.boundary_overlay_bgr != result.source_display_bgr)
    np.testing.assert_array_equal(image, original)


def test_float_nonfinite_and_unsupported_dtype_fail_clearly() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        run_thermal_segmentation(np.array([[0.0, np.nan]], dtype=np.float32))
    with pytest.raises(TypeError, match="dtype"):
        run_thermal_segmentation(np.ones((4, 4), dtype=np.int32))


def test_invalid_kernel_parameters_are_rejected() -> None:
    image, _ = _bright_fixture()
    with pytest.raises(ValueError, match="odd"):
        run_thermal_segmentation(image, config=ThermalPipelineConfig(gaussian_blur_kernel_size=2))


def test_no_ml_dependency_names_are_imported_by_module4_source() -> None:
    from pathlib import Path

    prohibited = {"torch", "tensorflow", "mediapipe", "transformers", "ultralytics", "detectron", "sam2"}
    source_root = Path(__file__).parents[1] / "src" / "module4"
    import_lines = []
    for path in source_root.rglob("*.py"):
        import_lines.extend(
            line.strip().lower()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip().startswith(("import ", "from "))
        )
    assert not any(package in line for line in import_lines for package in prohibited)
