# AGENTS.md — Module 4 (Human Boundary Detection: RGB & Thermal, + Fourier Theory)

Module-specific rules. This file **refines** `../AGENTS.md` and does not restate its general
Python / OpenCV / Git / documentation / web-app / integrity rules.

## Before doing anything in this module

1. Read `../AGENTS.md` and `../GIT_WORKFLOW.md` if reachable. If this module has been
   extracted into its own repository and the parent is not reachable, obtain the root rules
   another way before proceeding — they still fully apply.
2. Read, as the authoritative sources of truth for Module 4:
   - `MODULE_4_ASSIGNMENT_INSTRUCTIONS.md`
   - `MODULE_4_ASSIGNMENT_QUESTIONS.md`
3. Build a requirements checklist from those two documents before planning or writing code.

## What Module 4 requires

- **Question 1 — RGB human boundary detection:** find the exact boundary/contour of a human
  in a regular RGB image.
- **Question 2 — Thermal human boundary detection:** the same task on thermal/infrared
  imagery (a public thermal dataset may be used).
- **Question 3 — Theory:** a mathematical treatment of edge detection and region segmentation
  in the Fourier/frequency domain — 2D Fourier transform definition; meaning of low vs. high
  spatial frequencies, magnitude, phase; why edges correspond to high frequencies; high-pass
  filter derivation (`H_HP = 1 − H_LP`); the Fourier derivative property
  (`F{∂f/∂x} = j2πu·F(u,v)`); the frequency-domain Laplacian
  (`F{∇²f} = −4π²(u²+v²)·F(u,v)`); frequency-selective region segmentation; and
  local/windowed frequency analysis for texture-based segmentation. `MODULE_4_ASSIGNMENT_QUESTIONS.md`
  Parts A–F are the source of truth — do not work from this summary alone.

## Hard constraint — no ML/DL for Questions 1 and 2

**Machine-learning and deep-learning implementations are prohibited** for the required
Question 1 and Question 2 implementations. Do not use a pretrained detector or segmenter, a
neural network, a machine-learning classifier, or any foundation model to perform the
required segmentation, and do not let an ML/DL model hide inside a dependency.

The required implementation must use **traditional / classical image processing** via
OpenCV: color-space conversion, thresholding, adaptive/Otsu thresholding, background
subtraction, edge detection, morphological operations, connected-component analysis, contour
detection, region-based segmentation, region growing, and GrabCut where appropriate. Explain
and justify the chosen pipeline. The Question 3 theory must also stay non-ML.

## SAM2 — comparison/reference only

SAM2 (Segment Anything Model 2) is allowed **only as the comparison/reference segmentation**
described by the assignment. It must never replace the required classical OpenCV
implementation. Keep the **classical OpenCV result** and the **SAM2 result** clearly
separated in code, in saved outputs, and in the report. Never fabricate SAM2 masks or
metrics — if SAM2 has not actually been run, mark those outputs and numbers
**pending user experiment**.

## Expected outputs (per image, RGB and thermal)

Original image; preprocessing / intermediate results; detected human region; human
contour/boundary; final boundary visualization; SAM2 result; and the comparison between the
classical result and SAM2.

## Evaluation

Visual **and** numerical where possible: IoU, Dice, precision, recall, with `A` = the
classical OpenCV region and `B` = the reference (SAM2, or ground truth where available).
Provide the results table. Document the evaluation methodology accurately, including how the
masks were obtained and aligned. The RGB-vs-thermal discussion must report the behavior
**actually observed** in the experiments, not an assumption that one modality always wins.

## Web application surface

Separate RGB and thermal sections, each with: upload/select image, original, preprocessing
result, OpenCV segmentation, detected contour, SAM2 comparison, and evaluation metrics.

## Language note

`MODULE_4_ASSIGNMENT_INSTRUCTIONS.md` permits C/C++, Python, or MATLAB. Per the course
decision, implement in Python/OpenCV.

## Suggested task branches (see `../GIT_WORKFLOW.md`)

`task/module-4-rgb-human-boundary`, `task/module-4-thermal-human-boundary`,
`task/module-4-sam2-comparison`, `task/module-4-fourier-theory`.
