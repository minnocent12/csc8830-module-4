# Optional SAM2 reference comparison

This document records the implemented Phase 5 integration contract. It does not contain SAM2
metrics or claim that a checkpoint has been run. Any real inference remains pending user setup
and must be recorded from actual images and outputs.

## Official implementation and optional environment

The adapter follows Meta's official `facebookresearch/sam2` repository and image-predictor API:

- [official SAM2 repository](https://github.com/facebookresearch/sam2)
- [official README and installation requirements](https://github.com/facebookresearch/sam2/blob/main/README.md)
- [official installation notes](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md)
- [official image predictor source](https://github.com/facebookresearch/sam2/blob/main/sam2/sam2_image_predictor.py)
- [official checkpoint download script](https://github.com/facebookresearch/sam2/blob/main/checkpoints/download_ckpts.sh)
- [official Apache-2.0 license](https://github.com/facebookresearch/sam2/blob/main/LICENSE)
- [Meta research description](https://ai.meta.com/research/sam2/)

The current official README specifies Python 3.10 or newer and separate PyTorch/torchvision
requirements. It documents `build_sam2`, `SAM2ImagePredictor`, `set_image`, and box-prompt
`predict` usage. The Module 4 base environment does not add those dependencies. Use a separate
official checkout/environment, keep checkpoints outside the repository, and provide the local
path with `MODULE4_SAM2_CHECKPOINT` or the web-page field. Checkpoint files are ignored by the
repository.

The default UI configuration is SAM2.1 Hiera Tiny with
`configs/sam2.1/sam2.1_hiera_t.yaml`; the adapter does not download it. The official checkpoint
script provides the corresponding URL and filename. An implementation version may be entered
for provenance, but the adapter does not infer or fabricate one.

## Isolated data flow

The classical pipelines remain unchanged and do not import this adapter. The comparison page
always completes the classical RGB or thermal pipeline first. It then optionally runs SAM2 with a
separate user-supplied box prompt:

1. RGB source: OpenCV BGR `uint8` is explicitly converted to RGB HWC `uint8`.
2. Thermal source: single-channel or false-color thermal data is rendered to an explicit display
   RGB representation. This is a visual representation for SAM2, not calibrated temperature
   understanding.
3. The user enters SAM2 `x`, `y`, `width`, and `height`. The adapter converts this to official
   XYXY coordinates and passes `normalize_coords=True`.
4. The user may explicitly reuse the same user-supplied rectangle as the classical ROI. Otherwise
   the SAM2 box is independent. It is never derived from a classical mask, contour, component,
   threshold, polarity, or bounding box.
5. SAM2's predictor-native quality scores select one mask: highest score, first index on ties.
   IoU, Dice, or any reference metric is never used for selection.
6. The selected result must already be a binary mask and must match the source dimensions exactly.
   It is canonicalized to boolean using the Phase 4 validation rules; no silent resize occurs.
7. Only after this validation are comparison metrics computed against the classical mask. The
   label is `SAM2 reference segmentation`, never `ground truth`.

## Typed status and provenance

`module4.reference.sam2_reference` exposes an optional adapter and returns a
`SAM2ReferenceResult` with statuses `unavailable`, `pending`, `ready`, `completed`, or `failed`.
It includes the model/config, checkpoint identifier and source, implementation source/version,
device, prompt, source image identity/dimensions, selected index, predictor scores, selection
rule, warnings, and error. Incomplete states contain no dummy mask and produce no metrics.

The experiment runner also accepts reference status metadata and serializes it in JSON and CSV.
This permits a future real SAM2 result to remain traceable without changing the classical method
record or using the reference as a segmentation input.

## Web workflow and current evidence status

The Comparison and Evaluation page exposes separate uploaded-reference and SAM2-reference modes.
If SAM2, its checkpoint, or the requested device is unavailable, the page reports that state and
does not generate metrics. The code integration is implemented and covered by deterministic tests
without network, checkpoint, or GPU access. Real SAM2 inference is **pending**; no SAM2 masks,
timings, or empirical scores are included in this repository.

The official SAM2 repository is Apache-2.0 licensed; users must verify the applicable terms for
any checkpoint or redistributed artifact before submission. The adapter itself performs no
network download.
