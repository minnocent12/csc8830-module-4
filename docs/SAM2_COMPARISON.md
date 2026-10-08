# Comparison with SAM2

Questions 1 and 2 require the classical OpenCV results to be compared with SAM2 segmentation.
This document records that comparison for every fixed Phase 8 case: the official SAM2 runs,
visual and numerical results, measured processing cost, and a discussion of each comparison
criterion the assignment names. SAM2 is a deep-learning model, so it is the comparison method
only: it never feeds the classical pipelines, and it is not dataset ground truth. PyTorch and the
SAM2 checkpoint are kept out of the base install for the same reason; the app shows the recorded
SAM2 outputs when they are absent.

## Results at a glance

Means over three frames per image type, from `results/metrics/phase8_sam2_experiment_records.json`:

| Type | OpenCV vs ground truth IoU | SAM2 vs ground truth IoU | OpenCV vs SAM2 IoU |
|---|---:|---:|---:|
| RGB | 0.4228 | 0.6584 | 0.3186 |
| Thermal | 0.0277 | 0.6837 | 0.0220 |

SAM2's IoU against the ground truth is higher than the classical IoU on all six cases.
Per-case figures (original, classical boundary, SAM2 boundary, overlap) are
`results/comparisons/<case>_side_by_side.png`; the final report (Section 4) contains all six.

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

## Discussion by comparison criterion

Observed on the six side-by-side figures and the recorded metrics; the subset is three frames of
one indoor scene per modality, so none of this generalizes beyond it.

- **Boundary accuracy.** SAM2 traces the selected person closely (arms, the raised fist in 00185,
  head); its RGB precision against the ground truth is 0.92-0.99. The classical RGB boundary
  follows the body only where GrabCut separated it and otherwise runs along the box edges
  (precision against the ground truth 0.39-0.49).
- **Missing regions.** The classical RGB mask misses little (recall against the ground truth
  0.81-1.00). SAM2 misses whole people, not body parts: one box prompt yields one object, so the
  second, partly hidden, or seated person is absent (RGB recall against the ground truth
  0.61-0.79). This is a consequence of the single-box protocol; on thermal 00085 the same model
  segmented both people.
- **Extra/background regions.** The main classical RGB weakness: cabinet, door, wall, and table
  pixels inside the box remain foreground (precision against SAM2 about 0.31, recall about 0.98).
  SAM2 adds only small fragments: table specks (RGB 00135, 00185), blobs around the table
  (thermal 00135, 00185), and on the ceiling (thermal 00185).
- **Fine detail.** SAM2 keeps the hand holding a phone, the forearm-chest gap, and the leg
  separation. Classical morphology (3-pixel opening, 5-pixel closing) and GrabCut merge the arm
  with the table in 00135 and 00185. SAM2 leaves a speckled fragment at the hidden second
  person's head in RGB 00085.
- **Robustness to clutter.** Clutter inside the box is the dominant classical RGB failure. On
  thermal, the classical pipeline selected the dark Otsu hypothesis on all three frames, so its
  mask is mostly the cold background (mean IoU against the ground truth 0.028); SAM2 was largely
  unaffected.
- **Lighting sensitivity.** All frames share one scene and lighting, so this subset cannot
  measure it. The people are clearly brighter than the room in thermal, and SAM2 used that to reach
  its best thermal result (IoU 0.81 on 00085); the classical thermal failure is the polarity
  choice, not lighting.
- **Processing complexity.** See below.

## Processing complexity (measured)

`scripts/run_sam2_timing.py` reran both methods on the six cases on one Apple Silicon host
(macOS, Python 3.12.10, OpenCV 5.0.0, PyTorch 2.14.1): one warm-up, then five timed runs, median
reported. Full table: `results/metrics/phase8_timing_summary.md`.

| Method | Learned parameters | Weights | Per-image time |
|---|---:|---:|---|
| Classical thermal (Otsu, morphology) | 0 | none | about 4 ms (CPU) |
| Classical RGB (GrabCut, morphology) | 0 | none | 0.75-1.80 s (CPU) |
| SAM2.1 Hiera Tiny | 38,962,498 | 156 MB | about 107 ms (MPS), about 254 ms (CPU), plus a one-time model load of 0.16-0.48 s |

The rerun also verified reproducibility: every classical mask and every SAM2 mask (on both MPS
and CPU) was pixel-identical to the recorded evidence. PyTorch was 2.14.1 for the rerun versus
2.14.0 for the original SAM2 run.

## Web workflow

- **RGB and Thermal pages:** after the classical results, a "Comparison with SAM2" section shows
  the classical boundary, the SAM2 boundary, both together, the overlap, IoU/Dice/precision/recall
  of the classical mask against SAM2, the extra and missing pixel counts, and a three-way table
  against the dataset ground truth. All metrics are computed live from the classical mask just
  produced.
- **Comparison and Evaluation page:** the full six-case table and the measured timing first, then
  the interactive evaluation against SAM2 or an uploaded mask.
- **SAM2 source:** when the image is pixel-identical to a recorded dataset frame, the recorded
  official SAM2 mask is used (labeled with its model, prompt, and candidate). Otherwise SAM2 runs
  live if PyTorch, the `sam2` package, and `MODULE4_SAM2_CHECKPOINT` are available; if not, the
  page says so and shows no SAM2 mask or metric. The public deployment has no PyTorch, so it uses
  the recorded outputs for the bundled frames.

## Limitations and semantics

- SAM2 is a comparison/reference method and is not ground truth.
- The dataset ground-truth masks remain separate from SAM2 references.
- Thermal SAM2 results use false-color rendered inputs and do not measure physical temperature.
- The fixed six-case results are an empirical subset, not a generalization claim.
- No classical parameters were retuned after observing SAM2 results.
