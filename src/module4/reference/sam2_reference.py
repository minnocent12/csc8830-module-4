"""Optional, isolated adapter for the official Meta SAM 2 image predictor.

The adapter imports SAM2 and PyTorch lazily. The required classical application therefore remains
usable without either optional dependency, a checkpoint, or a supported accelerator. A completed
result contains only a canonical boolean mask and auditable metadata; model and tensor objects do
not cross this module's public boundary.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from module4.io_utils import validate_image_array
from module4.thermal import thermal_source_display_bgr
from module4.types import SAM2Config, SAM2Prompt, SAM2ReferenceResult, SAM2ReferenceStatus
from module4.validation import ReferenceValidationError, canonicalize_mask

OFFICIAL_SAM2_SOURCE = "https://github.com/facebookresearch/sam2"
DEFAULT_SELECTION_RULE = "highest predictor-provided SAM2 quality score; first index on ties"
CHECKPOINT_URLS = {
    "sam2.1_hiera_tiny": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    "sam2.1_hiera_small": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    "sam2.1_hiera_base_plus": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
    "sam2.1_hiera_large": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
}


class SAM2ReferenceError(ValueError):
    """Raised for invalid SAM2 prompt or output data."""


def _canonicalize_sam2_mask(mask: np.ndarray) -> np.ndarray:
    """Canonicalize an official SAM2 binary mask without thresholding logits.

    The official image predictor can return ``float32`` arrays containing exactly 0.0 and
    1.0 even when ``return_logits=False``. Those values are an explicit binary encoding, so
    they are converted to uint8 before passing through the project's strict mask validator.
    Arbitrary floating-point values remain rejected as logits or grayscale data.
    """
    if isinstance(mask, np.ndarray) and mask.dtype.kind == "f":
        if not np.isfinite(mask).all():
            raise SAM2ReferenceError("SAM2 mask contains non-finite values")
        values = np.unique(mask)
        if not np.isin(values, np.array([0.0, 1.0], dtype=mask.dtype)).all():
            raise SAM2ReferenceError(
                "SAM2 floating-point masks must be binary and contain only 0.0 and 1.0"
            )
        mask = mask.astype(np.uint8, copy=False)
    try:
        return canonicalize_mask(mask, name="SAM2 reference mask")
    except (TypeError, ValueError) as exc:
        raise SAM2ReferenceError(
            "SAM2 must return a binary bool/uint8 mask, or an exact float 0.0/1.0 mask, "
            "when return_logits=False"
        ) from exc


def validate_box_prompt(prompt: SAM2Prompt, image_shape: tuple[int, ...]) -> SAM2Prompt:
    """Validate an independently supplied XYWH box against an image shape."""
    if not isinstance(prompt, SAM2Prompt):
        raise TypeError("prompt must be a SAM2Prompt")
    if len(image_shape) < 2:
        raise ValueError("image_shape must contain height and width")
    height, width = int(image_shape[0]), int(image_shape[1])
    values = (prompt.x, prompt.y, prompt.width, prompt.height)
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in values):
        raise TypeError("SAM2 box coordinates and dimensions must be integers")
    if prompt.x < 0 or prompt.y < 0 or prompt.width <= 0 or prompt.height <= 0:
        raise ValueError("SAM2 box must have non-negative origin and positive dimensions")
    if prompt.x + prompt.width > width or prompt.y + prompt.height > height:
        raise ValueError("SAM2 box must lie fully inside the source image")
    return SAM2Prompt(int(prompt.x), int(prompt.y), int(prompt.width), int(prompt.height))


def bgr_to_sam2_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convert a copied OpenCV BGR uint8 image to the RGB HWC form expected by SAM2."""
    validate_image_array(image_bgr, name="SAM2 BGR image", color_order="BGR")
    if image_bgr.dtype != np.uint8 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("SAM2 image input requires an H x W x 3 uint8 BGR image")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def thermal_to_sam2_rgb(source_thermal: np.ndarray) -> np.ndarray:
    """Render thermal data explicitly as RGB appearance for SAM2, without implying temperature."""
    display_bgr = thermal_source_display_bgr(source_thermal)
    return bgr_to_sam2_rgb(display_bgr)


def select_sam2_mask(
    masks: np.ndarray,
    scores: np.ndarray,
    *,
    expected_shape: tuple[int, int],
) -> tuple[np.ndarray, int, tuple[float, ...]]:
    """Select the highest-scored official SAM2 binary mask without evaluation metrics."""
    mask_array = np.asarray(masks)
    if mask_array.ndim == 2:
        mask_array = mask_array[None, ...]
    if mask_array.ndim != 3 or mask_array.shape[0] == 0:
        raise SAM2ReferenceError("SAM2 predictor must return one or more C x H x W masks")
    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    if score_array.shape[0] != mask_array.shape[0] or not np.isfinite(score_array).all():
        raise SAM2ReferenceError("SAM2 predictor scores must be finite and match the mask count")
    index = int(np.argmax(score_array))
    selected = _canonicalize_sam2_mask(mask_array[index])
    if selected.shape != expected_shape:
        raise ReferenceValidationError(
            f"SAM2 reference mask shape {selected.shape} does not match source shape {expected_shape}"
        )
    return selected, index, tuple(float(value) for value in score_array)


def _checkpoint_identifier(config: SAM2Config) -> str | None:
    if not config.checkpoint_path:
        return None
    return Path(config.checkpoint_path).expanduser().name


def _base_provenance(config: SAM2Config, *, device: str | None = None) -> dict[str, str | int | float | None]:
    return {
        "implementation_source": config.implementation_source or OFFICIAL_SAM2_SOURCE,
        "implementation_version": config.implementation_version,
        "model_name": config.model_name,
        "model_config": config.model_config,
        "checkpoint_identifier": _checkpoint_identifier(config),
        "checkpoint_source": config.checkpoint_source or CHECKPOINT_URLS.get(config.model_name),
        "device": device,
    }


def _result(
    config: SAM2Config,
    *,
    status: SAM2ReferenceStatus,
    mask: np.ndarray | None = None,
    prompt: SAM2Prompt | None = None,
    source_image_id: str | None = None,
    source_dimensions: tuple[int, int] | None = None,
    device: str | None = None,
    selected_mask_index: int | None = None,
    predictor_scores: tuple[float, ...] = (),
    selection_rule: str | None = None,
    warnings: tuple[str, ...] = (),
    error: str | None = None,
) -> SAM2ReferenceResult:
    return SAM2ReferenceResult(
        status=status,
        mask=mask,
        model_name=config.model_name,
        model_config=config.model_config,
        checkpoint_identifier=_checkpoint_identifier(config),
        checkpoint_source=config.checkpoint_source or CHECKPOINT_URLS.get(config.model_name),
        implementation_source=config.implementation_source or OFFICIAL_SAM2_SOURCE,
        implementation_version=config.implementation_version,
        device=device,
        prompt_type="box" if prompt else None,
        prompt=prompt,
        source_image_id=source_image_id,
        source_dimensions=source_dimensions,
        selected_mask_index=selected_mask_index,
        predictor_scores=predictor_scores,
        selection_rule=selection_rule,
        provenance=_base_provenance(config, device=device),
        warnings=warnings,
        error=error,
    )


def pending_sam2_reference(
    *,
    config: SAM2Config = SAM2Config(),
    source_image_id: str | None = None,
    source_dimensions: tuple[int, int] | None = None,
) -> SAM2ReferenceResult:
    """Return an honest pending result without creating a placeholder mask."""
    return _result(
        config,
        status="pending",
        source_image_id=source_image_id,
        source_dimensions=source_dimensions,
        warnings=("SAM2 reference inference has not been run.",),
    )


def _load_sam2_dependencies() -> tuple[Any, Any, Any] | tuple[None, None, str]:
    try:
        import torch
    except ImportError as exc:
        return None, None, f"PyTorch is unavailable: {exc}"
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        return None, None, f"official SAM2 package is unavailable: {exc}"
    return torch, build_sam2, SAM2ImagePredictor


def _resolve_device(torch: Any, requested: str) -> str:
    allowed = {"auto", "cpu", "cuda", "mps"}
    if requested not in allowed:
        raise ValueError(f"device must be one of {sorted(allowed)}")
    if requested == "auto":
        if bool(torch.cuda.is_available()):
            return "cuda"
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and bool(mps.is_available()):
            return "mps"
        return "cpu"
    if requested == "cuda" and not bool(torch.cuda.is_available()):
        raise ValueError("CUDA was requested but is unavailable")
    if requested == "mps":
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is None or not bool(mps.is_available()):
            raise ValueError("Apple MPS was requested but is unavailable")
    return requested


@contextmanager
def _inference_context(torch: Any, device: str) -> Iterator[None]:
    with torch.inference_mode():
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                yield
        else:
            yield


def check_sam2_availability(config: SAM2Config = SAM2Config()) -> SAM2ReferenceResult:
    """Check optional dependency/checkpoint/device readiness without running inference."""
    if not config.checkpoint_path:
        return _result(config, status="unavailable", error="SAM2 checkpoint path was not supplied")
    checkpoint = Path(config.checkpoint_path).expanduser()
    if not checkpoint.is_file():
        return _result(config, status="unavailable", error=f"SAM2 checkpoint was not found: {checkpoint.name}")
    dependencies = _load_sam2_dependencies()
    if dependencies[0] is None:
        return _result(config, status="unavailable", error=dependencies[2])
    torch = dependencies[0]
    try:
        device = _resolve_device(torch, config.device)
    except (TypeError, ValueError) as exc:
        return _result(config, status="unavailable", error=str(exc))
    return _result(config, status="ready", device=device)


def run_sam2_reference(
    image_rgb: np.ndarray,
    prompt: SAM2Prompt,
    *,
    config: SAM2Config = SAM2Config(),
    source_image_id: str | None = None,
) -> SAM2ReferenceResult:
    """Run official SAM2 image prompting and return a provenance-bearing reference mask."""
    validate_image_array(image_rgb, name="SAM2 RGB image", color_order="RGB")
    if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("SAM2 RGB input requires an H x W x 3 uint8 RGB image")
    source_dimensions = (int(image_rgb.shape[0]), int(image_rgb.shape[1]))
    validated_prompt = validate_box_prompt(prompt, image_rgb.shape)
    readiness = check_sam2_availability(config)
    if readiness.status != "ready":
        return _result(
            config,
            status=readiness.status,
            prompt=validated_prompt,
            source_image_id=source_image_id,
            source_dimensions=source_dimensions,
            error=readiness.error,
            warnings=readiness.warnings,
        )
    device = readiness.device
    dependencies = _load_sam2_dependencies()
    if dependencies[0] is None:
        return _result(
            config,
            status="unavailable",
            prompt=validated_prompt,
            source_image_id=source_image_id,
            source_dimensions=source_dimensions,
            error=dependencies[2],
        )
    torch, build_sam2, predictor_class = dependencies
    try:
        model = build_sam2(config.model_config, str(Path(config.checkpoint_path).expanduser()), device=device, mode="eval")
        predictor = predictor_class(model)
        predictor.set_image(image_rgb.copy())
        box = np.asarray(validated_prompt.as_xyxy(), dtype=np.float32)
        with _inference_context(torch, device):
            masks, scores, _ = predictor.predict(
                box=box,
                multimask_output=True,
                return_logits=False,
                normalize_coords=True,
            )
        selected, index, score_values = select_sam2_mask(masks, scores, expected_shape=source_dimensions)
        return _result(
            config,
            status="completed",
            mask=selected,
            prompt=validated_prompt,
            source_image_id=source_image_id,
            source_dimensions=source_dimensions,
            device=device,
            selected_mask_index=index,
            predictor_scores=score_values,
            selection_rule=DEFAULT_SELECTION_RULE,
        )
    except (OSError, RuntimeError, TypeError, ValueError, ReferenceValidationError) as exc:
        return _result(
            config,
            status="failed",
            prompt=validated_prompt,
            source_image_id=source_image_id,
            source_dimensions=source_dimensions,
            device=device,
            error=f"SAM2 inference failed: {exc}",
        )
