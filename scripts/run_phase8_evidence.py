#!/usr/bin/env python3
"""Run the Phase 8 manifest and export traceable masks, overlays, and records.

The classical pipelines and metric formulas remain in ``module4``. This script only
coordinates the manifest, calls the existing experiment runner, and writes derived
evidence files under ``results/``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from module4.experiments import (  # noqa: E402
    run_experiment_case,
    write_csv_records,
    write_json_records,
)
from module4.io_utils import load_image_bgr, load_image_unchanged  # noqa: E402
from module4.metrics import evaluate_masks  # noqa: E402
from module4.rgb import run_rgb_segmentation  # noqa: E402
from module4.thermal import run_thermal_segmentation  # noqa: E402
from module4.types import RGBPipelineConfig, ROI, ThermalPipelineConfig  # noqa: E402
from module4.validation import prepare_reference  # noqa: E402
from module4.visualization import display_mask, mask_overlap_bgr  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="JSON manifest with explicit experiment cases")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Project root for relative paths")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Derived evidence output directory")
    return parser.parse_args()


def _roi(case: Mapping[str, Any]) -> ROI | None:
    value = case.get("roi")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("case roi must be an object with x, y, width, and height")
    return ROI(int(value["x"]), int(value["y"]), int(value["width"]), int(value["height"]))


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"could not write derived evidence image: {path}")


def _stable_artifact_path(path: Path, project_root: Path) -> str:
    """Return a project-relative path so committed metadata contains no local absolute path."""
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def _case_seed(case: Mapping[str, Any]) -> int:
    """Return a stable OpenCV RNG seed for one evidence case."""
    configured = case.get("rng_seed")
    if isinstance(configured, int) and not isinstance(configured, bool):
        return configured
    digest = hashlib.sha256(str(case["experiment_id"]).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def _run_classical(case: Mapping[str, Any], project_root: Path) -> tuple[np.ndarray, Any]:
    cv2.setRNGSeed(_case_seed(case))
    input_path = project_root / str(case["input_path"])
    roi = _roi(case)
    parameters = dict(case.get("parameters", {}))
    if case["modality"] == "rgb":
        if roi is None:
            raise ValueError("RGB evidence cases require a predefined ROI")
        source = load_image_bgr(input_path)
        result = run_rgb_segmentation(source, roi, config=RGBPipelineConfig(**parameters))
    elif case["modality"] == "thermal":
        source = load_image_unchanged(input_path)
        result = run_thermal_segmentation(source, roi, config=ThermalPipelineConfig(**parameters))
    else:
        raise ValueError("case modality must be rgb or thermal")
    return result.final_mask, result


def _reference_mask(case: Mapping[str, Any], prediction: np.ndarray, project_root: Path) -> np.ndarray | None:
    reference = case.get("reference")
    if not isinstance(reference, Mapping) or reference.get("status") not in {"available", "completed"}:
        return None
    reference_path = project_root / str(reference["path"])
    raw = load_image_unchanged(reference_path)
    prepared = prepare_reference(
        raw,
        expected_shape=prediction.shape,
        reference_status="available",
        reference_type=str(reference["type"]),
        image_id=str(case["image_id"]),
        reference_image_id=str(reference["image_id"]),
    )
    if prepared.mask is None:
        raise ValueError(f"reference validation returned no mask for {case['experiment_id']}")
    return prepared.mask


def _assert_metrics_match(record: Any, prediction: np.ndarray, reference: np.ndarray) -> None:
    measured = evaluate_masks(prediction, reference)
    if record.metrics is None:
        raise AssertionError(f"runner returned no metrics for completed case {record.image_id}")
    for name in ("iou", "dice", "precision", "recall"):
        if not math.isclose(float(getattr(record.metrics, name)), float(getattr(measured, name)), rel_tol=0, abs_tol=1e-12):
            raise AssertionError(f"runner/artifact metric mismatch for {record.image_id}: {name}")
    for name in ("tp", "fp", "fn", "tn"):
        if int(getattr(record.metrics, name)) != int(getattr(measured, name)):
            raise AssertionError(f"runner/artifact count mismatch for {record.image_id}: {name}")


def _format_metric(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_summary(records: list[Any], output_path: Path) -> None:
    """Serialize the generated records into a human-readable, reproducible summary."""
    rows = [record.to_dict() for record in records]
    lines = [
        "# Phase 8 experiment summary",
        "",
        "This table is generated by `scripts/run_phase8_evidence.py` from "
        "`phase8_experiment_records.json`; values are not manually entered.",
        "",
        "| Image ID | Modality | Method | Reference | Polarity | Status | IoU | Dice | Precision | Recall | TP | FP | FN | TN |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {image_id} | {modality} | {method} | {reference_type}/{reference_status} | {selected_polarity} | {status} | {iou} | {dice} | {precision} | {recall} | {tp} | {fp} | {fn} | {tn} |".format(
                selected_polarity=row.get("selected_polarity") or "-",
                **{name: _format_metric(row.get(name)) for name in (
                    "image_id", "modality", "method", "reference_type", "reference_status",
                    "status", "iou", "dice", "precision", "recall", "tp", "fp", "fn", "tn",
                )},
            )
        )
    lines.extend(["", "## Descriptive modality means", ""])
    for modality in ("rgb", "thermal"):
        modality_rows = [row for row in rows if row["modality"] == modality and row["status"] == "completed"]
        if not modality_rows:
            lines.append(f"- {modality}: no completed records.")
            continue
        means = {
            name: sum(float(row[name]) for row in modality_rows) / len(modality_rows)
            for name in ("iou", "dice", "precision", "recall")
        }
        lines.append(
            f"- {modality} (N={len(modality_rows)}): "
            + ", ".join(f"{name}={value:.6f}" for name, value in means.items())
            + "."
        )
    lines.extend([
        "",
        "These are descriptive means for the fixed three-frame Scene 1 subset, not a claim of generalization or modality superiority.",
        "",
    ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _export_case(case: Mapping[str, Any], record: Any, project_root: Path, output_dir: Path) -> dict[str, Any]:
    prediction, result = _run_classical(case, project_root)
    experiment_id = str(case["experiment_id"])
    modality = str(case["modality"])
    modality_dir = output_dir / modality
    _write_image(modality_dir / f"{experiment_id}_classical_mask.png", display_mask(prediction))
    _write_image(modality_dir / f"{experiment_id}_boundary_overlay.png", result.boundary_overlay_bgr)
    artifacts: dict[str, str] = {
        "classical_mask": _stable_artifact_path(modality_dir / f"{experiment_id}_classical_mask.png", project_root),
        "boundary_overlay": _stable_artifact_path(modality_dir / f"{experiment_id}_boundary_overlay.png", project_root),
    }
    if modality == "thermal":
        for name, image in {
            "normalized_intensity": result.normalized_intensity,
            "bright_candidate": display_mask(result.bright_cleaned_mask),
            "dark_candidate": display_mask(result.dark_cleaned_mask),
        }.items():
            path = modality_dir / f"{experiment_id}_{name}.png"
            _write_image(path, image)
            artifacts[name] = _stable_artifact_path(path, project_root)

    reference = _reference_mask(case, prediction, project_root)
    if reference is not None:
        _assert_metrics_match(record, prediction, reference)
        comparison_dir = output_dir / "comparisons"
        reference_path = comparison_dir / f"{experiment_id}_ground_truth_reference.png"
        overlap_path = comparison_dir / f"{experiment_id}_ground_truth_overlap.png"
        _write_image(reference_path, display_mask(reference))
        _write_image(overlap_path, mask_overlap_bgr(prediction, reference))
        artifacts["ground_truth_reference"] = _stable_artifact_path(reference_path, project_root)
        artifacts["ground_truth_overlap"] = _stable_artifact_path(overlap_path, project_root)
    return {
        "experiment_id": experiment_id,
        "image_id": record.image_id,
        "modality": modality,
        "status": record.status,
        "reference_type": record.reference_type,
        "reference_status": record.reference_status,
        "artifacts": artifacts,
    }


def main() -> int:
    args = _arguments()
    project_root = args.project_root.expanduser().resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else project_root / args.manifest
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    configuration = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(configuration, dict) or not isinstance(configuration.get("cases"), list):
        raise ValueError("manifest must be an object containing a cases array")

    cases = configuration["cases"]
    # GrabCut uses OpenCV's RNG internally. Seed each case independently so the serialized
    # metrics and exported masks describe the exact same pipeline invocation.
    records = []
    for case in cases:
        cv2.setRNGSeed(_case_seed(case))
        records.append(run_experiment_case(case, project_root=project_root))
    records = sorted(records, key=lambda record: (record.modality, record.image_id))
    metrics_dir = output_dir / "metrics"
    write_json_records(records, metrics_dir / "phase8_experiment_records.json")
    write_csv_records(records, metrics_dir / "phase8_experiment_records.csv")
    _write_summary(records, metrics_dir / "phase8_experiment_summary.md")
    artifact_records = []
    for record in records:
        matching_case = next(case for case in cases if case["image_id"] == record.image_id and case["modality"] == record.modality)
        artifact_records.append(_export_case(matching_case, record, project_root, output_dir))
    artifact_path = metrics_dir / "phase8_artifacts.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(sorted(artifact_records, key=lambda item: item["experiment_id"]), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "metrics": metrics_dir.as_posix(), "artifacts": artifact_path.as_posix()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
