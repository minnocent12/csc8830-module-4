#!/usr/bin/env python3
"""Measure processing cost of the classical pipelines and official SAM2 on the fixed cases.

This supplies the "processing complexity" comparison criterion. It reruns, on one host:

* the classical RGB/thermal pipeline for each fixed Phase 8 case, with the manifest
  parameters and recorded OpenCV RNG seed, and
* official SAM2 image prompting with the same predefined manifest box and model that
  produced the recorded SAM2 evidence.

Both reruns are checked against the frozen evidence masks so the timings describe the exact
computation that produced the reported comparison. Nothing under results/rgb, results/thermal,
or results/comparisons is overwritten; only the timing records below are written:

* results/metrics/phase8_timing_records.json
* results/metrics/phase8_timing_summary.md

Timing protocol: one untimed warm-up per case and stage, then ``--repeats`` timed runs using
``time.perf_counter``; the median is reported with min/max. SAM2 timing is split into model
load (once per device), image encoding (``set_image``), and box-prompt decoding (``predict``).
Accelerator work is synchronized before each clock read. Run inside the isolated SAM2
environment (PyTorch + official ``sam2`` package); the base Module 4 install cannot run it.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import cv2
import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from module4.io_utils import load_image_bgr, load_image_unchanged  # noqa: E402
from module4.metrics import evaluate_masks  # noqa: E402
from module4.reference.sam2_reference import (  # noqa: E402
    bgr_to_sam2_rgb,
    select_sam2_mask,
    thermal_to_sam2_rgb,
)
from module4.rgb import run_rgb_segmentation  # noqa: E402
from module4.thermal import run_thermal_segmentation  # noqa: E402
from module4.types import ROI, RGBPipelineConfig, SAM2Prompt, ThermalPipelineConfig  # noqa: E402
from module4.validation import canonicalize_mask  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=Path("data/experiment_manifest.json"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint", required=True, type=Path, help="Ignored local official SAM2 checkpoint")
    parser.add_argument("--model-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--devices", nargs="+", default=["mps", "cpu"], choices=("cpu", "mps", "cuda"))
    parser.add_argument("--repeats", type=int, default=5, help="Timed runs per case and stage (after one warm-up)")
    return parser.parse_args()


def _summary(samples_s: list[float]) -> dict[str, Any]:
    """Median/min/max in milliseconds plus the raw samples."""
    samples_ms = [value * 1000.0 for value in samples_s]
    return {
        "median_ms": statistics.median(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "samples_ms": samples_ms,
    }


def _time(function: Callable[[], Any], repeats: int, synchronize: Callable[[], None]) -> tuple[list[float], Any]:
    """Run once untimed (warm-up), then ``repeats`` timed runs; return samples and last output."""
    output = function()
    synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        output = function()
        synchronize()
        samples.append(time.perf_counter() - start)
    return samples, output


def _no_sync() -> None:
    return None


def _classical_runner(case: Mapping[str, Any], source: np.ndarray) -> Callable[[], np.ndarray]:
    roi_values = case["roi"]
    roi = ROI(*(int(roi_values[name]) for name in ("x", "y", "width", "height")))
    parameters = dict(case["parameters"])
    seed = int(case["rng_seed"])

    if case["modality"] == "rgb":
        config: RGBPipelineConfig | ThermalPipelineConfig = RGBPipelineConfig(**parameters)

        def run() -> np.ndarray:
            cv2.setRNGSeed(seed)  # GrabCut initialization uses the OpenCV RNG
            return run_rgb_segmentation(source, roi, config=config).final_mask
    else:
        config = ThermalPipelineConfig(**parameters)

        def run() -> np.ndarray:
            cv2.setRNGSeed(seed)
            return run_thermal_segmentation(source, roi, config=config).final_mask

    return run


def _host() -> dict[str, Any]:
    import torch

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "opencv_threads": cv2.getNumThreads(),
        "torch_threads": torch.get_num_threads(),
    }


def _write_summary(payload: Mapping[str, Any], path: Path) -> None:
    host = payload["host"]
    lines = [
        "# Phase 8 processing-cost comparison: classical OpenCV vs official SAM2",
        "",
        "Generated by `scripts/run_sam2_timing.py`; values are measured, not manually entered.",
        f"Host: {host['platform']} ({host['machine']}); Python {host['python']}; OpenCV {host['opencv']}; "
        f"PyTorch {host['torch']}.",
        f"Protocol: one warm-up then {payload['repeats']} timed runs per case and stage; median reported (ms).",
        "Classical runs on CPU through OpenCV. SAM2 time = image encoding (`set_image`) + box-prompt decoding (`predict`).",
        "",
        "## Model footprint",
        "",
        "| Method | Learned parameters | Weights on disk | Extra runtime dependencies |",
        "|---|---:|---:|---|",
        "| Classical OpenCV | 0 | 0 MB | none beyond OpenCV/NumPy |",
        f"| SAM2.1 Hiera Tiny | {payload['sam2_parameter_count']:,} | "
        f"{payload['checkpoint_size_bytes'] / 1e6:.1f} MB | PyTorch, official `sam2` package |",
        "",
        "## SAM2 one-time model load",
        "",
        "| Device | Load time (ms) |",
        "|---|---:|",
    ]
    for device, load in payload["sam2_model_load_ms"].items():
        lines.append(f"| {device} | {load:.1f} |")
    devices = list(payload["sam2_model_load_ms"])
    header = "| Case | Modality | Classical CPU (ms) | " + " | ".join(
        f"SAM2 {device} encode | SAM2 {device} decode | SAM2 {device} total" for device in devices
    ) + " |"
    lines.extend(["", "## Per-image median time (ms)", "", header, "|---|---|" + "---:|" * (1 + 3 * len(devices))])
    for case in payload["cases"]:
        cells = [case["experiment_id"], case["modality"], f"{case['classical']['median_ms']:.1f}"]
        for device in devices:
            sam2 = case["sam2"][device]
            cells.extend(
                [
                    f"{sam2['encode']['median_ms']:.1f}",
                    f"{sam2['decode']['median_ms']:.1f}",
                    f"{sam2['total_median_ms']:.1f}",
                ]
            )
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "## Reproduction check against frozen evidence", ""])
    lines.append("| Case | Classical rerun == frozen mask | " + " | ".join(
        f"SAM2 {device} rerun IoU vs recorded mask" for device in devices
    ) + " |")
    lines.append("|---|---|" + "---:|" * len(devices))
    for case in payload["cases"]:
        cells = [case["experiment_id"], str(case["classical_matches_frozen_mask"])]
        cells.extend(f"{case['sam2'][device]['rerun_iou_vs_recorded']:.6f}" for device in devices)
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "Timings are host-specific wall-clock measurements on this machine and are not a general",
            "benchmark. The fixed six-case subset is unchanged; no classical parameter was retuned.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = _arguments()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    project_root = args.project_root.expanduser().resolve()
    manifest = json.loads((project_root / args.manifest).read_text(encoding="utf-8"))
    cases = sorted(manifest["cases"], key=lambda case: (case["image_id"], case["modality"]))
    checkpoint = args.checkpoint.expanduser().resolve()

    def synchronizer(device: str) -> Callable[[], None]:
        if device == "mps":
            return torch.mps.synchronize
        if device == "cuda":
            return torch.cuda.synchronize
        return _no_sync

    # Load sources and run/time the classical pipeline once per case.
    case_rows: list[dict[str, Any]] = []
    sam2_inputs: dict[str, np.ndarray] = {}
    for case in cases:
        experiment_id = case["experiment_id"]
        source_path = project_root / case["input_path"]
        if case["modality"] == "rgb":
            source = load_image_bgr(source_path)
            sam2_inputs[experiment_id] = bgr_to_sam2_rgb(source)
        else:
            source = load_image_unchanged(source_path)
            sam2_inputs[experiment_id] = thermal_to_sam2_rgb(source)
        samples, classical_mask = _time(_classical_runner(case, source), args.repeats, _no_sync)
        frozen_path = project_root / "results" / case["modality"] / f"{experiment_id}_classical_mask.png"
        frozen = canonicalize_mask(load_image_unchanged(frozen_path), name="frozen classical mask")
        case_rows.append(
            {
                "experiment_id": experiment_id,
                "image_id": case["image_id"],
                "modality": case["modality"],
                "image_dimensions": list(source.shape[:2]),
                "prompt_xywh": [int(case["roi"][name]) for name in ("x", "y", "width", "height")],
                "classical": _summary(samples),
                "classical_matches_frozen_mask": bool(np.array_equal(classical_mask, frozen)),
                "sam2": {},
            }
        )

    load_ms: dict[str, float] = {}
    parameter_count = 0
    for device in args.devices:
        sync = synchronizer(device)
        start = time.perf_counter()
        model = build_sam2(args.model_config, str(checkpoint), device=device, mode="eval")
        predictor = SAM2ImagePredictor(model)
        sync()
        load_ms[device] = (time.perf_counter() - start) * 1000.0
        parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())

        for row in case_rows:
            image_rgb = sam2_inputs[row["experiment_id"]]
            prompt = SAM2Prompt(*row["prompt_xywh"])
            box = np.asarray(prompt.as_xyxy(), dtype=np.float32)

            def encode() -> None:
                with torch.inference_mode():
                    predictor.set_image(image_rgb.copy())

            def decode() -> tuple[Any, Any, Any]:
                with torch.inference_mode():
                    return predictor.predict(box=box, multimask_output=True, return_logits=False, normalize_coords=True)

            encode_samples, _ = _time(encode, args.repeats, sync)
            decode_samples, (masks, scores, _) = _time(decode, args.repeats, sync)
            mask, index, _ = select_sam2_mask(masks, scores, expected_shape=tuple(row["image_dimensions"]))
            recorded_path = project_root / "results" / row["modality"] / f"{row['experiment_id']}_sam2_mask.png"
            recorded = canonicalize_mask(load_image_unchanged(recorded_path), name="recorded SAM2 mask")
            encode_summary = _summary(encode_samples)
            decode_summary = _summary(decode_samples)
            row["sam2"][device] = {
                "encode": encode_summary,
                "decode": decode_summary,
                "total_median_ms": encode_summary["median_ms"] + decode_summary["median_ms"],
                "selected_mask_index": index,
                "rerun_iou_vs_recorded": evaluate_masks(mask, recorded).iou,
                "rerun_equals_recorded": bool(np.array_equal(mask, recorded)),
            }
        del predictor, model

    payload = {
        "description": "Measured processing cost of classical OpenCV pipelines and official SAM2 on the fixed Phase 8 cases.",
        "host": _host(),
        "repeats": args.repeats,
        "timer": "time.perf_counter, one warm-up per case and stage, accelerator synchronized",
        "sam2_model": {
            "model_config": args.model_config,
            "checkpoint_identifier": checkpoint.name,
            "implementation_version": manifest.get("sam2", {}).get("implementation_version"),
        },
        "sam2_parameter_count": parameter_count,
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "sam2_model_load_ms": load_ms,
        "cases": case_rows,
    }
    metrics_dir = project_root / "results" / "metrics"
    (metrics_dir / "phase8_timing_records.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_summary(payload, metrics_dir / "phase8_timing_summary.md")
    print((metrics_dir / "phase8_timing_summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
