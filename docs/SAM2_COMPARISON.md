# SAM2 reference comparison

This document records the genuine official SAM2 comparison run for the fixed Phase 8 evidence
set. SAM2 is a reference segmentation, never dataset ground truth, and it remains an optional
runtime for the classical application.

## Official implementation and runtime

- Official source: [facebookresearch/sam2](https://github.com/facebookresearch/sam2)
- Official checkout commit: `2b90b9f5ceec907a1c18123530e92e794ad901a4`
- Official image API: `build_sam2` and `SAM2ImagePredictor`
- Model: `sam2.1_hiera_tiny`
- Config: `configs/sam2.1/sam2.1_hiera_t.yaml`
- Checkpoint: `sam2.1_hiera_tiny.pt`
- Checkpoint source: `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt`
- Checkpoint SHA-256: `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`
- Checkpoint size: 156,008,466 bytes
- Inference Python: 3.12.10
- PyTorch: 2.14.0
- TorchVision: 0.29.0
- Device: Apple MPS; CUDA was unavailable and MPS was verified with a tensor operation

The official SAM2 base package was installed in an isolated environment with the CUDA extension
disabled because this host is Apple Silicon. The checkpoint is stored in the ignored local
`checkpoints/` directory and is not committed. The base Module 4 installation remains free of
PyTorch and SAM2 dependencies.

## Isolated data flow and prompt provenance

The classical pipelines remain unchanged. The exporter loads the already-generated Phase 8
classical masks and never uses a classical mask, contour, component, polarity, or metric to form
a SAM2 prompt or select a SAM2 candidate.

The six existing cases were retained exactly: Scene 1 frames `00085`, `00135`, and `00185`, in
both RGB and thermal modalities. For each case, the ROI recorded in the manifest was supplied as
the same independently predefined SAM2 XYWH box. The prompt source is recorded as:

`predefined manifest ROI supplied independently as a SAM2 box`

SAM2 returned three candidates per case. The selected candidate was always the highest
predictor-native score, with first-index tie handling. IoU, Dice, precision, recall, dataset
ground truth, and visual preference were not used for candidate selection.

RGB images were converted from OpenCV BGR to RGB. Thermal inputs were rendered through the
documented display representation and converted to RGB; this is a false-color visual input for
SAM2, not calibrated physical temperature data.

## Genuine fixed-case evidence

The reproducible exporter is `scripts/run_sam2_phase8_evidence.py`. It writes:

- `results/metrics/phase8_sam2_experiment_records.json`
- `results/metrics/phase8_sam2_experiment_records.csv`
- `results/metrics/phase8_sam2_experiment_summary.md`
- six SAM2 masks under `results/rgb/` and `results/thermal/`
- six classical-versus-SAM2 overlap images under `results/comparisons/`
- six SAM2-versus-dataset-ground-truth overlap images under `results/comparisons/`

The table reports actual pixel metrics. “Classical vs SAM2” compares the frozen Phase 8
classical mask to the selected SAM2 reference. “SAM2 vs dataset GT” independently compares the
SAM2 reference to the AAU VAP dataset mask.

| Frame | Modality | Native scores | Selected | Classical vs SAM2 IoU | SAM2 vs dataset GT IoU |
|---|---|---|---:|---:|---:|
| 00085 | RGB | 0.100976, 0.075563, 0.851205 | 2 | 0.308873 | 0.620474 |
| 00085 | Thermal | 0.623490, 0.558752, 0.722251 | 2 | 0.029198 | 0.811287 |
| 00135 | RGB | 0.369650, 0.250795, 0.461220 | 2 | 0.310382 | 0.581563 |
| 00135 | Thermal | 0.651283, 0.695221, 0.829059 | 2 | 0.016886 | 0.568167 |
| 00185 | RGB | 0.212998, 0.136643, 0.755984 | 2 | 0.336692 | 0.773312 |
| 00185 | Thermal | 0.609301, 0.722915, 0.758322 | 2 | 0.020001 | 0.671685 |

The JSON records contain the complete Dice, precision, recall, TP, FP, FN, TN, prompt,
dimensions, provenance, warnings, and artifact paths for every case.

## Web workflow

The Comparison and Evaluation page continues to run the classical pipeline first and then accepts
an optional independently supplied SAM2 box. It displays the real SAM2 provenance and metrics when
the official environment and local checkpoint are configured. Normal RGB, thermal, and Fourier
pages still run without SAM2 installed or a checkpoint at startup.

## Limitations and semantics

- SAM2 is a comparison/reference method and is not ground truth.
- The dataset ground-truth masks remain separate from SAM2 references.
- Thermal SAM2 results use false-color rendered inputs and do not measure physical temperature.
- The fixed six-case results are an empirical subset, not a generalization claim.
- No classical parameters were retuned after observing SAM2 results.
