from __future__ import annotations

import csv
import json

import cv2
import numpy as np

from module4.experiments import run_experiment_cases, run_experiment_case, write_csv_records, write_json_records
from module4.rgb import run_rgb_segmentation
from module4.thermal import run_thermal_segmentation
from module4.types import RGBPipelineConfig, ROI, ThermalPipelineConfig


def _write(path, image: np.ndarray) -> None:
    assert cv2.imwrite(str(path), image)


def test_runner_emits_pending_record_without_zero_metrics(tmp_path) -> None:
    image_path = tmp_path / "thermal.png"
    thermal = np.full((12, 12), 40, dtype=np.uint8)
    thermal[3:9, 4:8] = 220
    _write(image_path, thermal)

    record = run_experiment_case(
        {
            "image_id": "thermal-01",
            "modality": "thermal",
            "input_path": "thermal.png",
            "reference": {"status": "pending", "type": "ground_truth"},
        },
        project_root=tmp_path,
    )

    assert record.classical_status == "completed"
    assert record.status == "pending_reference"
    assert record.metrics is None
    assert record.to_dict()["iou"] is None
    assert "pending" in " ".join(record.warnings)


def test_runner_evaluates_available_reference_after_classical_processing(tmp_path) -> None:
    image_path = tmp_path / "thermal.png"
    reference_path = tmp_path / "reference.png"
    thermal = np.full((12, 12), 40, dtype=np.uint8)
    thermal[3:9, 4:8] = 220
    _write(image_path, thermal)
    prediction = run_thermal_segmentation(thermal, config=ThermalPipelineConfig()).final_mask
    _write(reference_path, np.where(prediction, 255, 0).astype(np.uint8))

    record = run_experiment_case(
        {
            "image_id": "thermal-01",
            "modality": "thermal",
            "input_path": "thermal.png",
            "reference": {
                "status": "available",
                "type": "ground_truth",
                "path": "reference.png",
                "image_id": "thermal-01",
            },
        },
        project_root=tmp_path,
    )

    assert record.status == "completed"
    assert record.reference_status == "available"
    assert record.metrics is not None
    assert record.metrics.iou == 1.0
    assert record.alignment is not None
    assert record.alignment.occurred is False
    assert record.source_image == "thermal.png"
    assert record.reference_image == "reference.png"


def test_runner_keeps_classical_configuration_independent_of_reference(tmp_path) -> None:
    image_path = tmp_path / "thermal.png"
    first_reference = tmp_path / "first.png"
    second_reference = tmp_path / "second.png"
    thermal = np.full((12, 12), 40, dtype=np.uint8)
    thermal[3:9, 4:8] = 220
    _write(image_path, thermal)
    _write(first_reference, np.zeros((12, 12), dtype=np.uint8))
    _write(second_reference, np.full((12, 12), 255, dtype=np.uint8))
    common = {"image_id": "thermal-01", "modality": "thermal", "input_path": "thermal.png"}

    first = run_experiment_case(
        {**common, "reference": {"status": "available", "type": "user_reference", "path": "first.png", "image_id": "thermal-01"}},
        project_root=tmp_path,
    )
    second = run_experiment_case(
        {**common, "reference": {"status": "available", "type": "user_reference", "path": "second.png", "image_id": "thermal-01"}},
        project_root=tmp_path,
    )

    assert first.processing_parameters == second.processing_parameters
    assert first.selected_polarity == second.selected_polarity
    assert first.metrics is not None and second.metrics is not None
    assert first.metrics.iou != second.metrics.iou


def test_runner_sorts_and_serializes_records_without_absolute_paths(tmp_path) -> None:
    rgb_path = tmp_path / "rgb.png"
    thermal_path = tmp_path / "thermal.png"
    rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    rgb[5:15, 7:13] = (40, 150, 220)
    _write(rgb_path, rgb)
    thermal = np.arange(144, dtype=np.uint8).reshape(12, 12)
    _write(thermal_path, thermal)
    cases = [
        {"image_id": "z-rgb", "modality": "rgb", "input_path": "rgb.png", "roi": [4, 4, 12, 12]},
        {"image_id": "a-thermal", "modality": "thermal", "input_path": "thermal.png"},
    ]

    records = run_experiment_cases(cases, project_root=tmp_path)
    json_path = tmp_path / "results" / "records.json"
    csv_path = tmp_path / "results" / "records.csv"
    write_json_records(records, json_path)
    write_csv_records(records, csv_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert [(item["modality"], item["image_id"]) for item in payload] == [("rgb", "z-rgb"), ("thermal", "a-thermal")]
    assert all(str(tmp_path) not in json.dumps(item) for item in payload)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["source_image"] == "rgb.png"
