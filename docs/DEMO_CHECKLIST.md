# Module 4 demonstration checklist

## Environment and evidence

- [ ] `python -m pytest -q` passes.
- [ ] `streamlit run app.py` starts from the Module 4 root.
- [ ] The fixed Phase 8 records and generated report are available.
- [ ] No raw dataset archive, checkpoint, secret, or private absolute path is visible.
- [ ] SAM2 is described as pending/unavailable if the official environment is not installed.

## RGB page

- [ ] RGB input is shown.
- [ ] User ROI is shown and described as an initialization input.
- [ ] GrabCut/intermediate stages are shown.
- [ ] Final mask and boundary overlay are shown.
- [ ] No reference mask is used as a segmentation input.

## Thermal page

- [ ] Thermal JPG is identified as false-color imagery, not calibrated temperature.
- [ ] Normalized intensity and both polarity candidates are shown.
- [ ] Selected polarity is stated.
- [ ] Warnings are not hidden.
- [ ] Final mask and boundary overlay are shown.

## Comparison page

- [ ] Classical prediction completes before reference evaluation.
- [ ] Binary reference validation and alignment status are shown.
- [ ] IoU, Dice, precision, recall, and confusion counts are explained.
- [ ] SAM2's independent prompt/reference semantics are explained.
- [ ] No unavailable SAM2 result is presented as a score or screenshot.

## Fourier page

- [ ] Parts A–F are visible or named.
- [ ] FFT magnitude/phase and centered spectrum are explained.
- [ ] Gaussian high-pass, derivative, and Laplacian behavior are shown.
- [ ] Part F's window/localization trade-off is stated.
- [ ] Synthetic Fourier content is labeled educational, not empirical.

## Closing

- [ ] State the fixed frames: 00085, 00135, 00185.
- [ ] State the scope: N=3 paired RGB/thermal subset, six records.
- [ ] State no case selection by score and no classical parameter retuning.
- [ ] State SAM2, video, and final submission packaging as pending where applicable.
