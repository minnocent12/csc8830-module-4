#!/usr/bin/env python3
"""Run official SAM2 on the fixed Phase 8 cases and export comparison evidence.

This script is intentionally separate from the classical Phase 8 exporter. It uses the
predefined manifest ROI as an independently documented SAM2 box prompt, loads the already
exported classical masks without rerunning or retuning those pipelines, and writes a second
set of records. SAM2 masks are references, never dataset ground truth.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from module4.io_utils import load_image_bgr, load_image_unchanged  # noqa: E402
from module4.metrics import evaluate_masks  # noqa: E402
from module4.reference.sam2_reference import (  # noqa: E402
    CHECKPOINT_URLS,
    bgr_to_sam2_rgb,
    run_sam2_reference,
    thermal_to_sam2_rgb,
)
from module4.types import SAM2Config, SAM2Prompt  # noqa: E402
from module4.validation import canonicalize_mask, prepare_reference  # noqa: E402
from module4.visualization import display_mask, mask_overlap_bgr  # noqa: E402

FIXED_EXPERIMENT_IDS = frozenset(
    {
        "aau-vap-scene1-00085-rgb",
        "aau-vap-scene1-00085-thermal",
        "aau-vap-scene1-00135-rgb",
        "aau-vap-scene1-00135-thermal",
        "aau-vap-scene1-00185-rgb",
        "aau-vap-scene1-00185-thermal",
    }
)
PROMPT_SOURCE = "predefined manifest ROI supplied independently as a SAM2 box"
SAM2_METHOD = "official_sam2_reference"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="Fixed Phase 8 manifest JSON")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--checkpoint", required=True, type=Path, help="Ignored local official SAM2 checkpoint")
    parser.add_argument("--model-name", default="sam2.1_hiera_tiny")
    parser.add_argument("--model-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--implementation-version", required=True, help="Official SAM2 git commit or release identifier")
    return parser.parse_args()


def _project_path(value: str | Path, project_root: Path) -> tuple[Path, str]:
    path = Path(value)
    resolved = path if path.is_absolute() else project_root / path
    absolute = resolved.expanduser().resolve()
    try:
        return absolute, absolute.relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"path must remain inside project root: {value}") from exc


def _roi(case: Mapping[str, Any]) -> tuple[int, int, int, int]:
    value = case.get("roi")
    if not isinstance(value, Mapping):
        raise ValueError(f"{case.get('experiment_id')} requires a mapping ROI")
    values = tuple(int(value[name]) for name in ("x", "y", "width", "height"))
    if values[2] <= 0 or values[3] <= 0:
        raise ValueError(f"{case.get('experiment_id')} ROI dimensions must be positive")
    return values


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"could not write evidence image: {path}")


def _stable_path(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def _metric_dict(metrics: Any) -> dict[str, Any]:
    return asdict(metrics)


def _assert_metric_match(actual: Any, recorded: Mapping[str, Any], label: str) -> None:
    for name in ("iou", "dice", "precision", "recall"):
        if not math.isclose(float(getattr(actual, name)), float(recorded[name]), rel_tol=0, abs_tol=1e-12):
            raise AssertionError(f"{label} metric mismatch: {name}")
    for name in ("tp", "fp", "fn", "tn"):
        if int(getattr(actual, name)) != int(recorded[name]):
            raise AssertionError(f"{label} count mismatch: {name}")


def _load_phase8_records(project_root: Path) -> dict[tuple[str, str], Mapping[str, Any]]:
    path = project_root / "results/metrics/phase8_experiment_records.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    result = {(str(row["image_id"]), str(row["modality"])): row for row in records}
    if len(result) != 6:
        raise ValueError("expected six existing Phase 8 records")
    return result


def _validate_fixed_cases(cases: list[Any]) -> None:
    ids = frozenset(str(case.get("experiment_id")) for case in cases if isinstance(case, Mapping))
    if ids != FIXED_EXPERIMENT_IDS or len(cases) != len(FIXED_EXPERIMENT_IDS):
        raise ValueError("manifest must contain exactly the six fixed Phase 8 RGB/thermal cases")


def _csv_row(record: Mapping[str, Any]) -> dict[str, Any]:
    sam2 = record["sam2"]
    prompt = record["prompt"]
    classical_vs_sam2 = record["comparisons"]["classical_vs_sam2"]
    sam2_vs_gt = record["comparisons"]["sam2_vs_dataset_ground_truth"]
    return {
        "experiment_id": record["experiment_id"],
        "image_id": record["image_id"],
        "modality": record["modality"],
        "method": record["method"],
        "source_image": record["source_image"],
        "source_dimensions": json.dumps(record["source_dimensions"]),
        "source_dtype": record["source_dtype"],
        "prompt_source": sam2["prompt_source"],
        "prompt_xywh": json.dumps(prompt["xywh"]),
        "prompt_xyxy": json.dumps(prompt["xyxy"]),
        "device": sam2["device"],
        "model_name": sam2["model_name"],
        "model_config": sam2["model_config"],
        "checkpoint_identifier": sam2["checkpoint_identifier"],
        "implementation_version": sam2["implementation_version"],
        "selected_mask_index": sam2["selected_mask_index"],
        "native_scores": json.dumps(sam2["native_scores"]),
        "mask_dimensions": json.dumps(sam2["mask_dimensions"]),
        "classical_vs_sam2": json.dumps(classical_vs_sam2, sort_keys=True),
        "sam2_vs_dataset_ground_truth": json.dumps(sam2_vs_gt, sort_keys=True),
        "status": record["status"],
        "warnings": json.dumps(record["warnings"]),
    }


def _write_records(records: list[dict[str, Any]], metrics_dir: Path) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "phase8_sam2_experiment_records.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    rows = [_csv_row(record) for record in records]
    with (metrics_dir / "phase8_sam2_experiment_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(records: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Phase 8 official SAM2 comparison evidence",
        "",
        "This file was generated by `scripts/run_sam2_phase8_evidence.py` from the fixed Phase 8 manifest.",
        "The six cases are unchanged: Scene 1 frames 00085, 00135, and 00185 in RGB and thermal.",
        "The predefined manifest ROI was supplied independently as the SAM2 box prompt.",
        "SAM2 candidates were selected only by the highest predictor-native score; dataset masks remain ground truth.",
        "",
        "| Image ID | Modality | Native scores | Selected | Classical vs SAM2 IoU | Dice | Precision | Recall | SAM2 vs GT IoU | Dice | Precision | Recall |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        sam2 = record["sam2"]
        first = record["comparisons"]["classical_vs_sam2"]
        second = record["comparisons"]["sam2_vs_dataset_ground_truth"]
        lines.append(
            "| {image_id} | {modality} | {scores} | {selected} | {a[iou]:.6f} | {a[dice]:.6f} | {a[precision]:.6f} | {a[recall]:.6f} | {b[iou]:.6f} | {b[dice]:.6f} | {b[precision]:.6f} | {b[recall]:.6f} |".format(
                image_id=record["image_id"],
                modality=record["modality"],
                scores=", ".join(f"{value:.6f}" for value in sam2["native_scores"]),
                selected=sam2["selected_mask_index"],
                a=first,
                b=second,
            )
        )
    lines.extend(
        [
            "",
            "The thermal source is a false-color rendered representation for SAM2; it is not calibrated temperature data.",
            "The classical masks and Phase 8 dataset-ground-truth metrics were loaded from existing frozen evidence and were not retuned.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = _arguments()
    project_root = args.project_root.expanduser().resolve()
    manifest_path, _ = _project_path(args.manifest, project_root)
    output_dir, _ = _project_path(args.output_dir, project_root)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    configuration = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = configuration.get("cases") if isinstance(configuration, Mapping) else None
    if not isinstance(cases, list):
        raise ValueError("manifest must contain a cases array")
    _validate_fixed_cases(cases)
    phase8_records = _load_phase8_records(project_root)
    checkpoint_source = CHECKPOINT_URLS.get(args.model_name)
    if checkpoint_source is None:
        raise ValueError(f"no official checkpoint URL is recorded for model {args.model_name}")
    config = SAM2Config(
        model_name=args.model_name,
        model_config=args.model_config,
        checkpoint_path=str(checkpoint_path),
        checkpoint_source=checkpoint_source,
        implementation_version=args.implementation_version,
        device=args.device,
    )

    records: list[dict[str, Any]] = []
    artifact_records: list[dict[str, Any]] = []
    for case in sorted(cases, key=lambda value: (str(value["image_id"]), str(value["modality"]))):
        experiment_id = str(case["experiment_id"])
        image_id = str(case["image_id"])
        modality = str(case["modality"])
        roi = _roi(case)
        prompt = SAM2Prompt(*roi)
        source_path, source_image = _project_path(str(case["input_path"]), project_root)
        if modality == "rgb":
            source = load_image_bgr(source_path)
            sam2_input = bgr_to_sam2_rgb(source)
            source_dtype = str(source.dtype)
        elif modality == "thermal":
            source = load_image_unchanged(source_path)
            sam2_input = thermal_to_sam2_rgb(source)
            source_dtype = str(source.dtype)
        else:
            raise ValueError(f"unsupported modality: {modality}")
        source_shape = (int(source.shape[0]), int(source.shape[1]))

        classical_path = project_root / "results" / modality / f"{experiment_id}_classical_mask.png"
        classical_mask = canonicalize_mask(load_image_unchanged(classical_path), name="frozen classical mask")
        if classical_mask.shape != source_shape:
            raise ValueError(f"frozen classical mask shape mismatch for {experiment_id}")

        reference = case.get("reference")
        if not isinstance(reference, Mapping) or reference.get("status") not in {"available", "completed"}:
            raise ValueError(f"dataset ground truth is required for {experiment_id}")
        gt_path, gt_relative = _project_path(str(reference["path"]), project_root)
        gt_raw = load_image_unchanged(gt_path)
        prepared = prepare_reference(
            gt_raw,
            expected_shape=source_shape,
            reference_status="available",
            reference_type="ground_truth",
            image_id=image_id,
            reference_image_id=str(reference["image_id"]),
        )
        if prepared.mask is None:
            raise ValueError(f"ground truth validation returned no mask for {experiment_id}")
        phase8 = phase8_records[(image_id, modality)]
        frozen_classical_vs_gt = evaluate_masks(classical_mask, prepared.mask)
        _assert_metric_match(frozen_classical_vs_gt, phase8, f"frozen Phase 8 {experiment_id}")

        sam2_result = run_sam2_reference(
            sam2_input,
            prompt,
            config=config,
            source_image_id=image_id,
        )
        if sam2_result.status != "completed" or sam2_result.mask is None:
            raise RuntimeError(f"SAM2 inference did not complete for {experiment_id}: {sam2_result.error}")
        sam2_mask = canonicalize_mask(sam2_result.mask, name="SAM2 reference mask")
        if sam2_mask.shape != source_shape:
            raise ValueError(f"SAM2 mask shape mismatch for {experiment_id}")

        classical_vs_sam2 = evaluate_masks(classical_mask, sam2_mask)
        sam2_vs_gt = evaluate_masks(sam2_mask, prepared.mask)
        modality_dir = output_dir / modality
        comparison_dir = output_dir / "comparisons"
        sam2_mask_path = modality_dir / f"{experiment_id}_sam2_mask.png"
        classical_vs_sam2_path = comparison_dir / f"{experiment_id}_classical_vs_sam2.png"
        sam2_vs_gt_path = comparison_dir / f"{experiment_id}_sam2_vs_ground_truth.png"
        _write_image(sam2_mask_path, display_mask(sam2_mask))
        _write_image(classical_vs_sam2_path, mask_overlap_bgr(classical_mask, sam2_mask))
        _write_image(sam2_vs_gt_path, mask_overlap_bgr(sam2_mask, prepared.mask))

        record = {
            "experiment_id": experiment_id,
            "image_id": image_id,
            "modality": modality,
            "method": SAM2_METHOD,
            "source_image": source_image,
            "source_dimensions": list(source_shape),
            "source_dtype": source_dtype,
            "reference_type": "sam2_reference",
            "reference_status": "completed",
            "reference_image": _stable_path(sam2_mask_path, project_root),
            "reference_dimensions": list(source_shape),
            "dataset_ground_truth": {
                "reference_type": "ground_truth",
                "reference_image": gt_relative,
                "reference_image_id": str(reference["image_id"]),
                "metrics_for_sam2": _metric_dict(sam2_vs_gt),
                "metrics_for_frozen_classical": _metric_dict(frozen_classical_vs_gt),
            },
            "prompt": {
                "source": PROMPT_SOURCE,
                "xywh": list(roi),
                "xyxy": list(prompt.as_xyxy()),
                "derived_from_classical_mask": False,
            },
            "sam2": {
                "implementation_source": config.implementation_source,
                "implementation_version": config.implementation_version,
                "model_name": sam2_result.model_name,
                "model_config": sam2_result.model_config,
                "checkpoint_identifier": sam2_result.checkpoint_identifier,
                "checkpoint_source": sam2_result.checkpoint_source,
                "device": sam2_result.device,
                "selected_mask_index": sam2_result.selected_mask_index,
                "native_scores": list(sam2_result.predictor_scores),
                "selection_rule": sam2_result.selection_rule,
                "mask_dimensions": list(source_shape),
                "mask_dtype": str(sam2_mask.dtype),
                "foreground_pixels": int(sam2_mask.sum()),
                "thermal_representation": (
                    "thermal_source_display_rgb; false-color display representation, not calibrated temperature"
                    if modality == "thermal"
                    else None
                ),
                "prompt_source": PROMPT_SOURCE,
            },
            "comparisons": {
                "classical_vs_sam2": _metric_dict(classical_vs_sam2),
                "sam2_vs_dataset_ground_truth": _metric_dict(sam2_vs_gt),
            },
            "frozen_classical_phase8_metrics": _metric_dict(frozen_classical_vs_gt),
            "artifacts": {
                "sam2_mask": _stable_path(sam2_mask_path, project_root),
                "classical_vs_sam2": _stable_path(classical_vs_sam2_path, project_root),
                "sam2_vs_ground_truth": _stable_path(sam2_vs_gt_path, project_root),
                "frozen_classical_mask": _stable_path(classical_path, project_root),
            },
            "warnings": [
                "SAM2 is a reference segmentation, not dataset ground truth.",
                *(["Thermal input was rendered as an RGB display representation; physical temperature was not inferred."] if modality == "thermal" else []),
            ],
            "status": "completed",
        }
        records.append(record)
        artifact_records.append({"experiment_id": experiment_id, "artifacts": record["artifacts"]})

    metrics_dir = output_dir / "metrics"
    _write_records(records, metrics_dir)
    _write_summary(records, metrics_dir / "phase8_sam2_experiment_summary.md")
    (metrics_dir / "phase8_sam2_artifacts.json").write_text(
        json.dumps(artifact_records, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"records": len(records), "metrics": str(metrics_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
