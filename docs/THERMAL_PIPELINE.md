# Classical Thermal Human Boundary Pipeline

This document describes the Module 4 Question 2 implementation. It is a classical,
ROI-optional thermal/intensity segmentation pipeline. It is not a learned person detector and
does not claim reliable automatic human detection.

## Scope and integrity

The implementation uses NumPy and OpenCV only: input validation, finite min/max normalization,
optional Gaussian denoising, dual-polarity Otsu thresholding, morphology, connected components,
deterministic scoring, external contours, and copy-based visualization. There is no ML/DL
dependency, SAM2 inference, Fourier implementation, dataset result, or fabricated experiment
claim in the Phase 3 pipeline. Phase 4 adds evaluation infrastructure around this pipeline. The
tracked Phase 8 evidence separately records provenance-verified RGB/thermal experiments and
official SAM2 reference comparisons; those results do not change this classical pipeline.

The pipeline should be described as **ROI-assisted classical human segmentation** when an ROI is
provided, or as **classical thermal intensity segmentation with transparent geometry heuristics**
when no ROI is provided. Neither mode is automatic human detection.

## Processing flow

```text
thermal source array
    ↓
validate shape, finite values, and supported dtype
    ↓
preserve source copy and derive a single-channel intensity array
    ↓
finite min/max normalization to uint8 [0, 255]
    ↓
optional Gaussian blur (disabled by kernel size 1)
    ↓
bright Otsu candidate: THRESH_BINARY + OTSU
dark Otsu candidate: THRESH_BINARY_INV + OTSU
    ↓
elliptical opening followed by closing on each candidate
    ↓
8-connected component analysis
    ↓
deterministic component scoring and polarity selection
    ↓
canonical bool final mask (False background, True foreground)
    ↓
RETR_EXTERNAL contour extraction
    ↓
boundary overlay on a display copy
```

## Supported input representations

- A 2D `uint8` array is treated as a single-channel intensity image.
- A 2D `uint16` array is treated as a higher-bit-depth intensity image. Its source dtype and
  intensity range are retained in the result metadata while a separate uint8 processing image
  is created.
- Finite `float32` and `float64` single-channel intensity arrays are also accepted; their values
  are not labeled as temperature units.
- A 3-channel array is treated explicitly as a false-color BGR palette representation. The
  pipeline computes `0.114 B + 0.587 G + 0.299 R` for intensity processing. This is a display/color
  conversion, not a conversion to Celsius, Fahrenheit, or calibrated radiometric temperature.
  The original three-channel array and its dtype remain available in the result.
- Empty arrays, non-finite floating-point values, unsupported dtypes, and unsupported channel
  counts fail clearly.

The source array is copied before processing. The normalized and enhanced arrays are separate
processing stages. For grayscale visualization, the normalized intensity is converted to BGR
only at the display boundary. For false-color input, display scaling is separate from the
computational intensity conversion.

## Normalization and optional enhancement

For finite single-channel intensity values `I`, let `a = min(I)` and `b = max(I)`. If `a != b`,
the processing image is:

```text
N = round(255 * (I - a) / (b - a))
```

clipped to `[0, 255]` and stored as `uint8`. A constant image maps to all zeros and is reported
as a constant failure status; it is never treated as a detected human. The values `a` and `b`
are returned as intensity metadata. They do not represent physical temperature without calibrated
source data.

The only optional enhancement is a small odd-sized Gaussian blur. Kernel size `1` disables it;
larger configured odd sizes use OpenCV `GaussianBlur` with `sigmaX=0`. Morphology remains a
separate later stage.

## Dual-polarity Otsu thresholding

Both interpretations are always computed from the normalized/enhanced uint8 image:

```python
bright = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
dark = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
```

The two threshold values and both raw boolean masks are retained. The implementation therefore
does not assume that a person is hotter/brighter than the background.

Each raw mask receives elliptical morphological opening followed by closing. Opening removes
small isolated foreground specks; closing fills small gaps. The raw and cleaned masks are kept
separate in the result.

## Component scoring and polarity selection

Every cleaned candidate is analyzed with 8-connected components. Components smaller than
`minimum_component_area` are excluded. For a component with area `A` in an image of area `T`,
relative area `r = A/T`, target area `q`, center-distance score `c`, border score `b`, and ROI
overlap fraction `o`, the transparent scores are:

```text
area_score       = min(r / q, 1) * (1 - r)
centrality_score = max(0, 1 - distance_to_image_center / max_corner_distance)
border_score     = 1 if the component touches no image border, otherwise 0
ROI score        = overlap_pixels / ROI_pixels, or 0 when no ROI is supplied
```

With an ROI, the component score is:

```text
0.60 * ROI score + 0.20 * area_score + 0.10 * centrality_score + 0.10 * border_score
```

Without an ROI, it is:

```text
0.50 * area_score + 0.35 * centrality_score + 0.15 * border_score
```

These are explainable geometry/area criteria, not a person-shape classifier. Components are
ranked by score descending, ROI overlap descending, area descending, and connected-component
label ascending. This makes selection reproducible and avoids selecting merely the candidate with
the most foreground pixels. Border contact is a penalty, not an unconditional rejection, so a
valid object touching an image edge remains possible.

The best bright and dark components are compared. A component must meet
`minimum_selection_score` to be viable. The viable component with the greater score is selected.
If both are viable and their score difference is at most `polarity_ambiguity_margin`, the higher
score still determines the output reproducibly, but the result status is `ambiguous` and a warning
is returned. If neither candidate is viable, the final mask is empty with status `empty`. Constant
images have status `constant`. A selected border-touching component adds a caution warning.

## Outputs and visualization

`ThermalSegmentationResult` preserves:

- the copied original source array and source metadata;
- the source-derived intensity, intensity range, normalized image, and enhancement stage;
- bright/dark Otsu thresholds and raw/cleaned masks;
- selected polarity, ambiguity flag, candidate component scores, and warnings;
- the canonical final `bool` mask;
- external contours as a separate output;
- a display BGR source copy and a non-destructive boundary overlay; and
- the exact numerical parameters used.

`False` is background and `True` is foreground in every computational mask. Display masks use
uint8 values 0 and 255. Contours do not replace the mask, and the overlay is drawn on a copy.

## Limitations and experiment scope

Otsu thresholding and simple component geometry are scene-dependent. The pipeline evaluates both
foreground polarities, but that does not prove that either candidate is a human. No reliability,
accuracy, robustness, or RGB-versus-thermal superiority claim is made here. The recorded Phase 8
experiment is limited to the fixed provenance-verified subset and its documented parameters.
SAM2 remains a separate reference segmentation rather than ground truth. Phase 4's metrics are
also available interactively when a user supplies a validated reference.
