"""Read access to the recorded official SAM2 comparison evidence.

The official SAM2 runs on the fixed AAU VAP cases are stored as JSON records plus binary mask
PNGs (see ``scripts/run_sam2_phase8_evidence.py``). This module only reads those files: it
imports no PyTorch or SAM2 code, so hosts without the deep-learning runtime (such as the public
web deployment) can still show the real SAM2 masks next to a live classical result.

A recorded mask is returned only when the caller's image is pixel-identical to the recorded
source image of the same modality. Matching is by content, never by file name, so an unrelated
upload can never be paired with a recorded SAM2 mask.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from module4.io_utils import load_image_bgr, load_image_unchanged
from module4.types import SAM2Prompt
from module4.validation import canonicalize_mask

SAM2_RECORDS = Path("results/metrics/phase8_sam2_experiment_records.json")
TIMING_RECORDS = Path("results/metrics/phase8_timing_records.json")


@dataclass(frozen=True)
class RecordedSAM2Case:
    """One recorded official SAM2 output and the provenance needed to label it."""

    experiment_id: str
    image_id: str
    modality: str
    prompt: SAM2Prompt
    mask: np.ndarray
    ground_truth_path: Path | None
    model_name: str
    implementation_version: str | None
    device: str | None
    selected_mask_index: int
    native_scores: tuple[float, ...]


def load_sam2_records(project_root: Path) -> list[dict[str, Any]]:
    """Return the recorded SAM2 comparison records, or an empty list if none are present."""
    path = project_root / SAM2_RECORDS
    if not path.is_file():
        return []
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("SAM2 records must be a JSON list")
    return sorted(records, key=lambda row: (str(row["image_id"]), str(row["modality"])))


def load_timing_records(project_root: Path) -> dict[str, Any] | None:
    """Return the measured classical-vs-SAM2 timing record, or None if it is absent."""
    path = project_root / TIMING_RECORDS
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or "cases" not in payload:
        raise ValueError("timing records must be a JSON object with a cases list")
    return dict(payload)


def _load_source(path: Path, modality: str) -> np.ndarray:
    # Same decoders as the pages and the exporter: BGR for RGB, unchanged for thermal.
    return load_image_bgr(path) if modality == "rgb" else load_image_unchanged(path)


def find_recorded_sam2_case(image: np.ndarray, modality: str, project_root: Path) -> RecordedSAM2Case | None:
    """Return the recorded SAM2 case whose source image equals ``image`` exactly, if any."""
    if modality not in {"rgb", "thermal"}:
        raise ValueError("modality must be 'rgb' or 'thermal'")
    for record in load_sam2_records(project_root):
        if record.get("modality") != modality or record.get("status") != "completed":
            continue
        source_path = project_root / str(record["source_image"])
        mask_path = project_root / str(record["reference_image"])
        if not source_path.is_file() or not mask_path.is_file():
            continue
        source = _load_source(source_path, modality)
        if source.shape != image.shape or source.dtype != image.dtype or not np.array_equal(source, image):
            continue
        mask = canonicalize_mask(load_image_unchanged(mask_path), name="recorded SAM2 mask")
        ground_truth = record.get("dataset_ground_truth") or {}
        ground_truth_path = project_root / str(ground_truth["reference_image"]) if ground_truth.get("reference_image") else None
        sam2 = record["sam2"]
        return RecordedSAM2Case(
            experiment_id=str(record["experiment_id"]),
            image_id=str(record["image_id"]),
            modality=modality,
            prompt=SAM2Prompt(*(int(value) for value in record["prompt"]["xywh"])),
            mask=mask,
            ground_truth_path=ground_truth_path if ground_truth_path and ground_truth_path.is_file() else None,
            model_name=str(sam2["model_name"]),
            implementation_version=sam2.get("implementation_version"),
            device=sam2.get("device"),
            selected_mask_index=int(sam2["selected_mask_index"]),
            native_scores=tuple(float(value) for value in sam2["native_scores"]),
        )
    return None
