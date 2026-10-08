#!/usr/bin/env python3
"""Build one side-by-side classical-vs-SAM2 figure per fixed case from committed evidence.

For each recorded SAM2 case this reads the source image, the frozen classical mask, the recorded
official SAM2 mask, and the AAU VAP ground-truth mask, and writes
``results/comparisons/<experiment_id>_side_by_side.png`` with four panels:

1. original image (thermal shown through the documented display rendering),
2. classical OpenCV boundary (red),
3. official SAM2 boundary (cyan),
4. classical-vs-SAM2 pixel overlap: green = both, red = classical only, blue = SAM2 only.

The panel title reports IoU values recomputed here from the masks (not copied from the
records). No model is run; this script needs only the base Module 4 environment + Matplotlib.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from module4.contours import draw_contours_on_bgr, extract_external_contours  # noqa: E402
from module4.io_utils import load_image_bgr, load_image_unchanged  # noqa: E402
from module4.metrics import evaluate_masks  # noqa: E402
from module4.reference.recorded import load_sam2_records  # noqa: E402
from module4.thermal import thermal_source_display_bgr  # noqa: E402
from module4.validation import canonicalize_mask  # noqa: E402
from module4.visualization import (  # noqa: E402
    CLASSICAL_BOUNDARY_BGR,
    SAM2_BOUNDARY_BGR,
    bgr_to_rgb,
    mask_overlap_bgr,
)


def _mask(path: Path, name: str):
    return canonicalize_mask(load_image_unchanged(path), name=name)


def build_figures(project_root: Path) -> list[Path]:
    written: list[Path] = []
    for record in load_sam2_records(project_root):
        modality = record["modality"]
        source_path = project_root / record["source_image"]
        display_bgr = (
            load_image_bgr(source_path)
            if modality == "rgb"
            else thermal_source_display_bgr(load_image_unchanged(source_path))
        )
        classical = _mask(project_root / record["artifacts"]["frozen_classical_mask"], "classical mask")
        sam2 = _mask(project_root / record["artifacts"]["sam2_mask"], "SAM2 mask")
        truth = _mask(project_root / record["dataset_ground_truth"]["reference_image"], "ground truth")

        classical_overlay = draw_contours_on_bgr(
            display_bgr, extract_external_contours(classical), color=CLASSICAL_BOUNDARY_BGR, thickness=3
        )
        sam2_overlay = draw_contours_on_bgr(
            display_bgr, extract_external_contours(sam2), color=SAM2_BOUNDARY_BGR, thickness=3
        )
        panels = [
            (bgr_to_rgb(display_bgr), "Original" if modality == "rgb" else "Original (thermal display)"),
            (bgr_to_rgb(classical_overlay), "Classical OpenCV boundary"),
            (bgr_to_rgb(sam2_overlay), "SAM2 boundary"),
            (bgr_to_rgb(mask_overlap_bgr(classical, sam2)), "Overlap: green both, red OpenCV only,\nblue SAM2 only"),
        ]
        vs_sam2 = evaluate_masks(classical, sam2).iou
        classical_gt = evaluate_masks(classical, truth).iou
        sam2_gt = evaluate_masks(sam2, truth).iou
        frame = record["image_id"].rsplit("-", 1)[-1]

        figure, axes = plt.subplots(1, 4, figsize=(16, 3.6), constrained_layout=True)
        for axis, (image, title) in zip(axes, panels):
            axis.imshow(image)
            axis.set_title(title, fontsize=10)
            axis.axis("off")
        figure.suptitle(
            f"Frame {frame} {modality.upper()}: IoU OpenCV vs SAM2 = {vs_sam2:.3f} | "
            f"OpenCV vs ground truth = {classical_gt:.3f} | SAM2 vs ground truth = {sam2_gt:.3f}",
            fontsize=11,
        )
        output = project_root / "results" / "comparisons" / f"{record['experiment_id']}_side_by_side.png"
        figure.savefig(output, dpi=110)
        plt.close(figure)
        written.append(output)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    for path in build_figures(args.project_root.expanduser().resolve()):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
