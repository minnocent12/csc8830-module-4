# Phase 4 evaluation and experiment protocol

This document describes implemented evaluation infrastructure, not completed academic
experiments. No input images, reference masks, metric values, timing values, or RGB-versus-thermal
conclusions are included in the repository. The user must collect and verify those materials.

## Evaluation contract

`module4.metrics.evaluate_masks(prediction, reference)` accepts two aligned 2D masks and returns
one `SegmentationMetrics` record containing pixel-level IoU, Dice, precision, recall, and TP/FP/
FN/TN counts. Both inputs are copied into the canonical boolean convention: `False` is background
and `True` is foreground. External masks are accepted only when they are boolean or `uint8` with
values from `{0, 1, 255}`. Arbitrary grayscale values are rejected instead of implicitly
thresholded.

The metric edge conventions are:

- IoU = `TP / (TP + FP + FN)`, and IoU is `1.0` when both masks have empty foreground;
- Dice = `2 TP / (2 TP + FP + FN)`, and Dice is `1.0` when both masks have empty foreground;
- precision = `TP / (TP + FP)` and recall = `TP / (TP + FN)`;
- if both masks have empty foreground, precision and recall are `1.0`; otherwise a zero
  denominator produces `0.0`.

Shape mismatch always fails. No resize is implicit. If a mismatch is justified by the experiment
design, call `align_reference_mask` explicitly. It accepts nearest-neighbor interpolation only
and records original dimensions, destination dimensions, transformation, interpolation, and
whether alignment occurred. Identity and orientation metadata are checked before comparison.

## Reference semantics

The allowed reference types are `ground_truth`, `sam2_reference`, and `user_reference`. They are
provenance labels, not interchangeable inference methods. Phase 4 records could label a
user-supplied mask as `sam2_reference`; Phase 5 adds an isolated optional official SAM2 adapter.
A completed Phase 5 SAM2 row records the model, config, checkpoint identifier, device, prompt,
native-score selection rule, and source identity. SAM2 is still a reference segmentation, not
ground truth.

The prediction image ID and reference image ID must match. An explicitly mismatched orientation
also fails. A missing/pending reference yields a pending record with null metrics and a warning;
it is not a zero-valued result. A failed reference is recorded as failed with null metrics.

## Reproducible runner

`module4.experiments.run_experiment_cases` accepts explicit case mappings and reuses
`module4.rgb`, `module4.thermal`, `module4.metrics`, and `module4.validation`. It does not
duplicate either segmentation pipeline. Classical processing runs to completion before the
reference path is loaded, so reference masks cannot influence the prediction. Each record stores
the image ID, modality, method, project-relative input/reference paths, source dimensions and
dtype, parameters, ROI/polarity, reference status/type, reference metadata, alignment metadata, metrics, status, and
warnings. Absolute paths outside the configured project root are rejected.

Example configuration (illustrative schema only; it is not an experiment row):

```json
{
  "cases": [
    {
      "image_id": "rgb-frame-001",
      "modality": "rgb",
      "input_path": "data/rgb/frame-001.png",
      "roi": [120, 80, 240, 480],
      "parameters": {"grabcut_iterations": 5},
      "reference": {
        "status": "pending",
        "type": "ground_truth"
      }
    },
    {
      "image_id": "thermal-frame-001",
      "modality": "thermal",
      "input_path": "data/thermal/frame-001.tiff",
      "reference": {
        "status": "available",
        "type": "user_reference",
        "path": "data/references/frame-001.png",
        "image_id": "thermal-frame-001",
        "alignment": "resize_nearest"
      }
    }
  ]
}
```

Run it with:

```text
python scripts/run_experiments.py --config path/to/config.json --project-root . \
  --output-json results/experiment_records.json --output-csv results/experiment_records.csv
```

The example's pending row is intentionally not stored as data. Replace paths, IDs, and reference
status only after the corresponding user data exists.

## User experiment procedure

1. Collect RGB and/or thermal images and preserve source provenance, dimensions, and dtype.
2. Run the standalone classical RGB or thermal page and record the exact ROI and parameters.
3. Create or obtain a reference mask for the same image ID. Verify its foreground convention and
   allowed binary encoding.
4. Run the experiment runner or Comparison and Evaluation page. Use explicit nearest-neighbor
   alignment only when the acquisition/annotation process justifies it.
5. Inspect prediction, reference, and TP/FP/FN overlap images. Record observations as pending
   until they are actually observed and saved.
6. Transfer only generated, traceable records into the final report. Automated tests validate
   implementation; they do not count as physical or empirical validation.

## Empty report table

Populate this table only from generated records after the user experiment. Empty cells are
intentional and must not be replaced with plausible values.

| Image ID | Modality | Method | Reference type | Status | IoU | Dice | Precision | Recall | TP | FP | FN | TN | Notes |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pending user data | RGB/Thermal | classical pipeline | pending | pending |  |  |  |  |  |  |  |  |  |

Phase 5 status: the isolated SAM2 reference integration and comparison workflow is implemented.
Real checkpoint inference and empirical SAM2 comparison results remain pending until the user
supplies the separate official environment, local checkpoint, and actual images. Phase 6 remains
out of scope here; its only recommendation is the Fourier Parts A-F theory work.
