# Module 4 demonstration script

Target duration: 8–10 minutes. This script describes the existing application and tracked
evidence. Record only outputs actually shown by the running app; do not create SAM2 output if the
optional official environment is unavailable.

## Before recording

1. Activate the documented virtual environment and run `python -m pytest -q`.
2. Start the standalone app with `streamlit run app.py`.
3. Have one RGB image, one thermal image, and matching reference masks available locally.
4. Keep the fixed Phase 8 evidence records and report open for the results discussion.
5. If SAM2 is unavailable, show the explicit unavailable/pending status rather than attempting to
   imply an inference result.

## 1. Introduction — about 45 seconds

State that Module 4 covers classical RGB and thermal human-boundary detection, strict reference
evaluation, an optional SAM2 reference adapter, and Fourier Parts A–F. Explain that references are
used after prediction for evaluation and that the real-data claims are limited to six fixed AAU VAP
cases.

## 2. RGB Human Boundary page — about 2 minutes

1. Open `RGB Human Boundary`.
2. Upload/select the RGB image and point out the explicit BGR-to-RGB/display handling.
3. Enter or draw the ROI; say that the ROI is a user input, not a ground-truth mask.
4. Run the classical pipeline.
5. Show the ROI/initialization, GrabCut result, morphology/component stage, and final boundary.
6. Explain that the final mask is boolean and that parameters are visible and reproducible.

## 3. Thermal Human Boundary page — about 2 minutes

1. Open `Thermal Human Boundary` and upload/select the thermal JPG.
2. Explain that this dataset input is false-color three-channel imagery, not calibrated temperature.
3. Run the pipeline and show normalized intensity, bright and dark candidates, the selected mask,
   and the boundary overlay.
4. Point out the selected polarity and any border-touching or false-color warnings.

## 4. Comparison and Evaluation — about 1.5 minutes

1. Open `Comparison and Evaluation`.
2. Run a classical prediction first.
3. Select an uploaded binary reference mask and show the validation/alignment information.
4. Show IoU, Dice, precision, recall, and the TP/FP/FN/TN interpretation.
5. Explain that the optional SAM2 mode uses an independent box prompt and that SAM2 is a reference
   segmentation, never ground truth.
6. If the official package/checkpoint is unavailable, show the pending/unavailable message and do
   not report a mask, timing, or metric.

## 5. Fourier Theory — about 1.5 minutes

Open `Fourier Theory` and briefly show Parts A–F:

- Part A: 2D FFT, magnitude, phase, and centered display;
- Part B: why rapid intensity transitions require high frequencies;
- Part C: Gaussian high-pass filtering;
- Part D: derivative multipliers `j 2 pi u` and `j 2 pi v`;
- Part E: Laplacian multiplier `-4 pi^2 (u^2 + v^2)`;
- Part F: frequency-band energy and the localization trade-off of windowed analysis.

Label the synthetic Fourier example as educational theory, not real experimental evidence.

## 6. Results and limitations — about 1 minute

Show the report's generated result table and state: three fixed frames (00085, 00135, 00185),
both RGB and thermal, six records total, no case selected by score, and no classical retuning.
Report the descriptive subset means from the table only. State that the AAU VAP thermal images are
false-color and that real SAM2 inference remains blocked in the current environment.

## Closing statement

The application exposes all assignment components, the experiment records are machine-readable,
and the report distinguishes implemented, tested, experimentally validated, and pending claims.
The final submission still requires a reviewed report, a recorded video, and any instructor-facing
packaging required by the course.
