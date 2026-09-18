"""Typed experiment records and orchestration for real or pending user-configured cases."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import cv2

from module4.io_utils import load_image_bgr, load_image_unchanged
from module4.metrics import evaluate_masks
from module4.rgb import run_rgb_segmentation
from module4.thermal import run_thermal_segmentation
from module4.types import (
    AlignmentMetadata,
    RGBPipelineConfig,
    ReferenceType,
    ROI,
    SegmentationMetrics,
    ThermalPipelineConfig,
)
from module4.validation import (
    ReferenceValidationError,
    align_reference_mask,
    prepare_reference,
)

Modality = Literal["rgb", "thermal"]
ExperimentReferenceStatus = Literal["available", "pending", "unavailable", "ready", "completed", "failed"]
ExperimentStatus = Literal["completed", "pending_reference", "reference_unavailable", "failed"]


@dataclass(frozen=True)
class ExperimentRecord:
    """One reproducible classical-segmentation case and its optional evaluation."""

    image_id: str
    modality: Modality
    method: str
    source_image: str
    source_dimensions: tuple[int, int] | None
    source_dtype: str | None
    reference_type: ReferenceType | None
    reference_status: ExperimentReferenceStatus
    reference_image: str | None
    reference_dimensions: tuple[int, int] | None
    processing_parameters: Mapping[str, Any]
    roi: tuple[int, int, int, int] | None
    selected_polarity: str | None
    metrics: SegmentationMetrics | None
    alignment: AlignmentMetadata | None
    classical_status: Literal["completed", "failed"]
    status: ExperimentStatus
    warnings: tuple[str, ...]
    reference_metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible record with stable field names."""
        values: dict[str, Any] = {
            "image_id": self.image_id,
            "modality": self.modality,
            "method": self.method,
            "source_image": self.source_image,
            "source_dimensions": list(self.source_dimensions) if self.source_dimensions else None,
            "source_dtype": self.source_dtype,
            "reference_type": self.reference_type,
            "reference_status": self.reference_status,
            "reference_image": self.reference_image,
            "reference_dimensions": list(self.reference_dimensions) if self.reference_dimensions else None,
            "reference_metadata": dict(self.reference_metadata) if self.reference_metadata else None,
            "processing_parameters": dict(self.processing_parameters),
            "roi": list(self.roi) if self.roi else None,
            "selected_polarity": self.selected_polarity,
            "alignment": asdict(self.alignment) if self.alignment else None,
            "classical_status": self.classical_status,
            "status": self.status,
            "warnings": list(self.warnings),
        }
        if self.metrics is None:
            values.update({name: None for name in METRIC_FIELDS})
        else:
            values.update(asdict(self.metrics))
        return values


METRIC_FIELDS = (
    "iou",
    "dice",
    "precision",
    "recall",
    "tp",
    "fp",
    "fn",
    "tn",
    "prediction_foreground_pixels",
    "reference_foreground_pixels",
)
CSV_FIELDS = (
    "image_id",
    "modality",
    "method",
    "source_image",
    "source_dimensions",
    "source_dtype",
    "reference_type",
    "reference_status",
    "reference_image",
    "reference_dimensions",
    "reference_metadata",
    "roi",
    "selected_polarity",
    "alignment",
    "classical_status",
    "status",
    *METRIC_FIELDS,
    "processing_parameters",
    "warnings",
)


def _stable_path(path: Path, project_root: Path) -> str:
    resolved = path.expanduser().resolve()
    root = project_root.expanduser().resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("input and reference paths must be inside the project root") from exc


def _resolve_path(value: object, project_root: Path, *, name: str) -> tuple[Path, str]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty project-relative path")
    path = Path(value)
    resolved = path if path.is_absolute() else project_root / path
    return resolved, _stable_path(resolved, project_root)


def _coerce_roi(value: object) -> ROI | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        try:
            return ROI(int(value["x"]), int(value["y"]), int(value["width"]), int(value["height"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("roi mapping must contain x, y, width, and height") from exc
    if isinstance(value, (str, bytes)):
        raise TypeError("roi must be a four-value sequence or mapping")
    try:
        values = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("roi must be a four-value sequence or mapping") from exc
    if len(values) != 4:
        raise ValueError("roi must contain x, y, width, and height")
    if not all(isinstance(item, int) for item in values):
        raise TypeError("roi values must be integers")
    return ROI(*(int(item) for item in values))


def _parameters(case: Mapping[str, Any]) -> dict[str, Any]:
    parameters = case.get("parameters", {})
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must be an object")
    return dict(parameters)


def _record(
    *,
    image_id: str,
    modality: Modality,
    method: str,
    source_image: str,
    source_dimensions: tuple[int, int] | None,
    source_dtype: str | None,
    reference_type: ReferenceType | None,
    reference_status: ExperimentReferenceStatus,
    reference_image: str | None,
    reference_dimensions: tuple[int, int] | None,
    processing_parameters: Mapping[str, Any],
    roi: ROI | None,
    selected_polarity: str | None,
    metrics: SegmentationMetrics | None,
    alignment: AlignmentMetadata | None,
    classical_status: Literal["completed", "failed"],
    status: ExperimentStatus,
    warnings: Sequence[str],
    reference_metadata: Mapping[str, Any] | None = None,
) -> ExperimentRecord:
    return ExperimentRecord(
        image_id=image_id,
        modality=modality,
        method=method,
        source_image=source_image,
        source_dimensions=source_dimensions,
        source_dtype=source_dtype,
        reference_type=reference_type,
        reference_status=reference_status,
        reference_image=reference_image,
        reference_dimensions=reference_dimensions,
        processing_parameters=dict(processing_parameters),
        roi=roi.as_tuple() if roi else None,
        selected_polarity=selected_polarity,
        metrics=metrics,
        alignment=alignment,
        classical_status=classical_status,
        status=status,
        warnings=tuple(warnings),
        reference_metadata=dict(reference_metadata) if reference_metadata else None,
    )


def _reference_spec(case: Mapping[str, Any]) -> Mapping[str, Any]:
    value = case.get("reference")
    if value is None:
        return {"status": "pending", "type": None}
    if not isinstance(value, Mapping):
        raise TypeError("reference must be an object")
    return value


def _evaluate_reference(
    *,
    case: Mapping[str, Any],
    result_mask: Any,
    image_id: str,
    project_root: Path,
    base_warnings: list[str],
) -> tuple[ExperimentReferenceStatus, ReferenceType | None, str | None, tuple[int, int] | None, SegmentationMetrics | None, AlignmentMetadata | None, ExperimentStatus, list[str], Mapping[str, Any] | None]:
    """Evaluate only after the classical pipeline has completed."""
    spec = _reference_spec(case)
    raw_status = spec.get("status")
    reference_path = spec.get("path")
    if raw_status in {"available", "completed"}:
        status: ExperimentReferenceStatus = "available"
    elif raw_status in {"pending", "unavailable", "ready", "failed"}:
        status = raw_status
    else:
        status = "available" if reference_path else "pending"
    reference_type = spec.get("type")
    if reference_type not in {"ground_truth", "sam2_reference", "user_reference", None}:
        raise ValueError("reference.type must be ground_truth, sam2_reference, or user_reference")
    reference_image = None
    if reference_path is not None:
        _, reference_image = _resolve_path(reference_path, project_root, name="reference.path")
    warnings = list(base_warnings)
    reference_metadata = spec.get("metadata") if isinstance(spec.get("metadata"), Mapping) else None
    if status == "pending":
        warnings.append("Reference mask is pending; no metrics were generated.")
        return "pending", reference_type, reference_image, None, None, None, "pending_reference", warnings, reference_metadata
    if status == "unavailable":
        warnings.append("Reference integration is unavailable; no metrics were generated.")
        return "unavailable", reference_type, reference_image, None, None, None, "reference_unavailable", warnings, reference_metadata
    if status == "ready":
        warnings.append("Reference integration is ready but inference has not completed; no metrics were generated.")
        return "ready", reference_type, reference_image, None, None, None, "reference_unavailable", warnings, reference_metadata
    if status == "failed":
        warnings.append("Reference status is failed; no metrics were generated.")
        return "failed", reference_type, reference_image, None, None, None, "failed", warnings, reference_metadata
    if reference_path is None:
        raise ReferenceValidationError("available reference requires reference.path")
    reference_path_resolved, _ = _resolve_path(reference_path, project_root, name="reference.path")
    reference = load_image_unchanged(reference_path_resolved)
    reference_image_id = spec.get("image_id")
    if not isinstance(reference_image_id, str) or not reference_image_id:
        raise ReferenceValidationError("available reference requires an explicit reference.image_id")
    if reference_type is None:
        raise ReferenceValidationError("available reference requires an explicit reference.type")
    orientation = spec.get("orientation")
    prediction_orientation = case.get("orientation")
    alignment_mode = spec.get("alignment")
    if alignment_mode not in {None, "resize_nearest"}:
        raise ValueError("reference.alignment must be resize_nearest when supplied")
    if alignment_mode == "resize_nearest":
        canonical, alignment = align_reference_mask(
            reference,
            result_mask.shape,
            prediction_image_id=image_id,
            reference_image_id=reference_image_id,
            reference_type=reference_type,
            prediction_orientation=prediction_orientation if isinstance(prediction_orientation, str) else None,
            reference_orientation=orientation if isinstance(orientation, str) else None,
        )
    else:
        prepared = prepare_reference(
            reference,
            expected_shape=result_mask.shape,
            reference_status="available",
            reference_type=reference_type,
            image_id=image_id,
            reference_image_id=reference_image_id,
            prediction_orientation=prediction_orientation if isinstance(prediction_orientation, str) else None,
            reference_orientation=orientation if isinstance(orientation, str) else None,
        )
        canonical = prepared.mask
        alignment = prepared.alignment
    if canonical is None:
        raise ReferenceValidationError("reference validation returned no mask")
    metrics = evaluate_masks(result_mask, canonical)
    warnings.extend([])
    return "available", reference_type, reference_image, (int(reference.shape[0]), int(reference.shape[1])), metrics, alignment, "completed", warnings, reference_metadata


def run_experiment_case(case: Mapping[str, Any], *, project_root: Path) -> ExperimentRecord:
    """Run one configured RGB or thermal case without allowing reference leakage."""
    if not isinstance(case, Mapping):
        raise TypeError("each experiment case must be an object")
    image_id = case.get("image_id")
    modality = case.get("modality")
    if not isinstance(image_id, str) or not image_id:
        raise ValueError("image_id must be a non-empty string")
    if modality not in {"rgb", "thermal"}:
        raise ValueError("modality must be rgb or thermal")
    modality = modality  # type: ignore[assignment]
    source_path, source_image = _resolve_path(case.get("input_path"), project_root, name="input_path")
    roi = _coerce_roi(case.get("roi"))
    parameters = _parameters(case)
    rng_seed = case.get("rng_seed")
    if rng_seed is not None:
        if isinstance(rng_seed, bool) or not isinstance(rng_seed, int) or not 0 <= rng_seed <= 0x7FFFFFFF:
            raise ValueError("rng_seed must be an integer between 0 and 2147483647")
        cv2.setRNGSeed(rng_seed)
    method = "classical_rgb_grabcut" if modality == "rgb" else "classical_thermal_otsu"
    warnings: list[str] = []
    try:
        if modality == "rgb":
            if roi is None:
                raise ValueError("RGB cases require an ROI")
            source = load_image_bgr(source_path)
            result = run_rgb_segmentation(source, roi, config=RGBPipelineConfig(**parameters))
            source_dimensions = tuple(int(value) for value in result.final_mask.shape)
            source_dtype = str(result.original_bgr.dtype)
            processing_parameters: dict[str, Any] = dict(result.parameters)
            selected_polarity = None
            result_mask = result.final_mask
        else:
            source = load_image_unchanged(source_path)
            result = run_thermal_segmentation(source, roi, config=ThermalPipelineConfig(**parameters))
            source_dimensions = tuple(int(value) for value in result.final_mask.shape)
            source_dtype = str(result.original_thermal.dtype)
            processing_parameters = dict(result.parameters)
            processing_parameters.update(
                {
                    "bright_otsu_threshold": result.bright_otsu_threshold,
                    "dark_otsu_threshold": result.dark_otsu_threshold,
                    "input_representation": result.input_representation,
                }
            )
            selected_polarity = result.selected_polarity
            result_mask = result.final_mask
        if rng_seed is not None:
            processing_parameters["rng_seed"] = rng_seed
        warnings.extend(result.warnings)
        reference_status, reference_type, reference_image, reference_dimensions, metrics, alignment, status, warnings, reference_metadata = _evaluate_reference(
            case=case,
            result_mask=result_mask,
            image_id=image_id,
            project_root=project_root,
            base_warnings=warnings,
        )
        return _record(
            image_id=image_id,
            modality=modality,
            method=method,
            source_image=source_image,
            source_dimensions=source_dimensions,
            source_dtype=source_dtype,
            reference_type=reference_type,
            reference_status=reference_status,
            reference_image=reference_image,
            reference_dimensions=reference_dimensions,
            processing_parameters=processing_parameters,
            roi=roi,
            selected_polarity=selected_polarity,
            metrics=metrics,
            alignment=alignment,
            classical_status="completed",
            status=status,
            warnings=warnings,
            reference_metadata=reference_metadata,
        )
    except (OSError, TypeError, ValueError) as exc:
        try:
            reference_spec = _reference_spec(case)
        except (TypeError, ValueError):
            reference_spec = {}
        reference_type = reference_spec.get("type") if reference_spec.get("type") in {"ground_truth", "sam2_reference", "user_reference"} else None
        reference_status = reference_spec.get("status") if reference_spec.get("status") in {"available", "pending", "unavailable", "ready", "completed", "failed"} else ("available" if reference_spec.get("path") else "pending")
        if reference_status in {"available", "completed"}:
            reference_status = "failed"
        reference_metadata = reference_spec.get("metadata") if isinstance(reference_spec.get("metadata"), Mapping) else None
        return _record(
            image_id=image_id,
            modality=modality,
            method=method,
            source_image=source_image,
            source_dimensions=None,
            source_dtype=None,
            reference_type=reference_type,
            reference_status=reference_status,
            reference_image=None,
            reference_dimensions=None,
            processing_parameters=parameters,
            roi=roi,
            selected_polarity=None,
            metrics=None,
            alignment=None,
            classical_status="failed",
            status="failed",
            warnings=(f"Case failed: {exc}",),
            reference_metadata=reference_metadata,
        )


def run_experiment_cases(cases: Sequence[Mapping[str, Any]], *, project_root: Path) -> list[ExperimentRecord]:
    """Run configured cases and return deterministic image/modality ordering."""
    records = [run_experiment_case(case, project_root=project_root) for case in cases]
    return sorted(records, key=lambda record: (record.modality, record.image_id))


def write_json_records(records: Sequence[ExperimentRecord], output_path: Path) -> None:
    """Write deterministic JSON records with project-relative paths only."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [record.to_dict() for record in sorted(records, key=lambda item: (item.modality, item.image_id))]
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv_records(records: Sequence[ExperimentRecord], output_path: Path) -> None:
    """Write a compact deterministic CSV suitable for later result tables."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for record in sorted(records, key=lambda item: (item.modality, item.image_id)):
            value = record.to_dict()
            row = {
                field: value.get(field)
                for field in CSV_FIELDS
            }
            for field in ("source_dimensions", "reference_dimensions", "reference_metadata", "roi", "alignment", "processing_parameters", "warnings"):
                row[field] = json.dumps(row[field], sort_keys=True, separators=(",", ":"))
            writer.writerow(row)
