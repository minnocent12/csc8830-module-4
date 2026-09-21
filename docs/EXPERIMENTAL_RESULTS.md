# Phase 8 experimental results and protocol

This document records the controlled Phase 8 experiment run. Claims are classified explicitly:
the runner and artifact exporter are **implemented**; the automated tests and artifact/metric
consistency checks are **tested**; the six fixed AAU VAP cases below are **experimentally
validated** only for this dataset subset and procedure; the six official SAM2 reference runs are
**experimentally validated** only for this fixed prompt/model/device procedure.
No generalization, timing, or RGB-versus-thermal superiority claim is made.

## Dataset, provenance, and selection

The experiment uses three predefined Scene 1 frame IDs from the AAU VAP Trimodal People
Segmentation Dataset, version 3, acquired through the public Kaggle distribution. The dataset
and its source are documented at [Kaggle](https://www.kaggle.com/datasets/aalborguniversity/trimodal-people-segmentation)
and the [official AAU project page](https://vap.aau.dk/vap-trimodal-people-segmentation-dataset/).
The listed dataset terms are CC BY 4.0. The cited source is Palmero et al. (2016),
“Multi-modal RGB-Depth-Thermal Human Body Segmentation,” *International Journal of Computer
Vision*, 118(2), 217–239.

The deterministic selection rule was fixed before metric computation: use frame IDs **00085,
00135, and 00185**, in ascending order, with both RGB and thermal modalities for each frame.
No case was selected or excluded using a reference mask, metric, or score. The source-only ROIs
are recorded in [`data/experiment_manifest.json`](../data/experiment_manifest.json). The raw
downloaded files, derived binary references, and checkpoints remain ignored local data; only
small derived evidence and serialized records are tracked.

Thermal inputs are the dataset's three-channel false-color JPG representations. They are decoded
as OpenCV BGR and converted to an intensity image for the existing classical thermal pipeline;
they are not calibrated temperature arrays.

## Phase 8 cases

| Frame | RGB ROI (x, y, width, height) | Thermal ROI | Reference |
|---|---|---|---|
| 00085 | (120, 20, 450, 460) | (120, 20, 450, 460) | aligned dataset person mask |
| 00135 | (80, 80, 350, 400) | (80, 80, 350, 400) | aligned dataset person mask |
| 00185 | (50, 60, 350, 420) | (50, 60, 350, 420) | aligned dataset person mask |

The dataset's person-label values were canonicalized as a derived binary reference: zero maps to
background and any documented nonzero person label maps to foreground. The conversion metadata
and source values are in the manifest. Classical processing completed before any reference was
loaded, so the references could not influence prediction, polarity selection, or component
selection.

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

## Re-running the experiment

Use the fixed manifest and evidence exporter from the repository root:

```text
python scripts/run_phase8_evidence.py \
  --manifest data/experiment_manifest.json \
  --project-root . \
  --output-dir results
```

The manifest records an independent `rng_seed` for each case because the existing GrabCut
operation uses OpenCV RNG internally. The exporter applies that seed before each invocation and
then verifies that the exported masks produce the same metrics and confusion counts serialized by
the runner.

## Generated Phase 8 evidence

The serialized records and generated summary are authoritative for the values below:

- [`results/metrics/phase8_experiment_records.json`](../results/metrics/phase8_experiment_records.json)
- [`results/metrics/phase8_experiment_records.csv`](../results/metrics/phase8_experiment_records.csv)
- [`results/metrics/phase8_experiment_summary.md`](../results/metrics/phase8_experiment_summary.md)
- [`results/metrics/phase8_artifacts.json`](../results/metrics/phase8_artifacts.json)
- per-case masks, boundary overlays, thermal intermediates, references, and TP/FP/FN overlays
  under [`results/`](../results/)

The summary table is generated by `scripts/run_phase8_evidence.py`; it is not manually entered.
The same script re-evaluates each exported mask against its canonical reference and asserts that
all four metrics and four confusion counts match the runner record.

## Observations from this fixed subset

The RGB pipeline completed on all three cases with no pipeline warnings. The thermal pipeline
selected the dark polarity on all three cases and emitted the documented false-color/intensity
warning plus a border-touching-component caution on each case. These are observations of the
stored records, not broad performance claims. The summary's descriptive means are provided only
to describe this N=3 subset; they should not be reported as dataset-wide estimates.

## Official SAM2 reference status

The official SAM2 source is [facebookresearch/sam2](https://github.com/facebookresearch/sam2).
Genuine inference was completed with the official checkout at commit
`2b90b9f5ceec907a1c18123530e92e794ad901a4`, SAM2.1 Hiera Tiny, config
`configs/sam2.1/sam2.1_hiera_t.yaml`, and the official `sam2.1_hiera_tiny.pt` checkpoint. The
isolated runtime used Python 3.12.10, PyTorch 2.14.0, TorchVision 0.29.0, and Apple MPS. CUDA
was unavailable; MPS was verified before inference.

The fixed prompt strategy used the manifest ROI as an independently predefined SAM2 box for all
six existing cases: frames 00085, 00135, and 00185 in RGB and thermal. The three returned masks
were selected using the highest SAM2-native predictor score only. No classical mask, metric,
dataset ground-truth mask, or visual preference influenced prompt or candidate selection.

Additional SAM2 evidence is serialized in:

- `results/metrics/phase8_sam2_experiment_records.json`
- `results/metrics/phase8_sam2_experiment_records.csv`
- `results/metrics/phase8_sam2_experiment_summary.md`

The existing Phase 8 dataset-ground-truth records remain unchanged. SAM2 masks are labeled
`sam2_reference`, not `ground_truth`. Thermal SAM2 input is a rendered false-color RGB
representation and does not imply calibrated temperature understanding. Complete provenance,
native scores, prompts, masks, comparisons, and metrics are in the SAM2 records and
`docs/SAM2_COMPARISON.md`.

Phase 5's isolated SAM2 adapter, Phase 6's Fourier Parts A–F theory, and Phase 7's Streamlit
integration remain implemented/tested components. They are not substitutes for Phase 8's real
SAM2 comparison or for a final academic report.
