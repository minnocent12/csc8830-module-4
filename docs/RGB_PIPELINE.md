# Question 1: ROI-Assisted Classical RGB Human Boundary

## Scope and constraint

Question 1 requires a regular color image, a human region, its boundary/contour, intermediate
processing outputs, a final visualization, and a later comparison with SAM2. The required
segmentation here is classical OpenCV only. No machine-learning or deep-learning detector,
classifier, neural network, or pretrained model selects the ROI or changes the mask.

The current implementation is deliberately described as **ROI-assisted classical human
segmentation**, not automatic human detection. The user supplies a rectangle around the
intended person. This makes the initialization explicit and reproducible for a still image
without hiding a learned person detector inside the pipeline.

## Input and mask conventions

The core function accepts an OpenCV BGR image with shape H x W x 3 and dtype uint8. It does not
silently treat BGR as RGB. Conversion to RGB occurs only at the display boundary.

The ROI is strict xywh:

- x, y: upper-left pixel coordinate;
- width, height: positive pixel dimensions.

The ROI must be fully inside the image and each side must be at least two pixels. Partial
out-of-bounds rectangles are rejected rather than silently clamped. The final internal mask is
a new boolean array: False is background and True is foreground. Display/export masks use 0 and
255 uint8 values.

## Processing path

1. Validate the BGR uint8 image and ROI.
2. Create grayscale, HSV, and Lab diagnostics. These are labeled diagnostics, not hidden
   detection models or additional learned stages.
3. Initialize a GrabCut label image as definite background.
4. Run OpenCV GrabCut with the user rectangle and GC_INIT_WITH_RECT. The default is five
   iterations.
5. Convert GC_FGD and GC_PR_FGD to True. GC_BGD and GC_PR_BGD become False. Raw integer
   GrabCut labels and the canonical boolean mask remain separate outputs.
6. Apply elliptical morphological opening, then closing. Opening removes small isolated
   foreground specks; closing fills small holes and connects small boundary gaps. Defaults are
   3 x 3 opening and 5 x 5 closing kernels. The default minimum connected-component area is
   16 pixels and the default contour thickness is 2 pixels.
7. Compute 8-connected components. Keep components meeting the minimum area and overlapping
   the user ROI.
8. Select the component with greatest ROI overlap, then greatest area, then lowest component
   label as a deterministic tie-break. Ties on overlap and area produce a warning.
9. Extract external contours from the final binary mask using RETR_EXTERNAL and
   CHAIN_APPROX_SIMPLE.
10. Draw the contour and ROI on copies of the original BGR image.

The result object preserves the original copy, ROI overlay, diagnostics, raw labels, raw
foreground candidate, cleaned candidate, selected component, final mask, contour tuple,
boundary overlay, parameters, and warnings.

## Failure and limitation behavior

Malformed, too-small, negative, or partially out-of-bounds ROIs raise validation errors.
If no component meets both the minimum-area and ROI-overlap rules, the result contains an empty
final mask and a warning rather than claiming that a person was found. If multiple components
tie under the documented rule, the selected label is reproducible and the ambiguity is exposed.

GrabCut quality depends on the user's ROI, color contrast, clothing/background similarity,
lighting, image resolution, and boundary texture. This implementation does not claim robust
automatic performance on arbitrary RGB scenes.

## Current evaluation status

Pixel-level metrics and conservative reference validation are implemented separately in
`module4.metrics` and `module4.validation` during Phase 4. This document still reports no
empirical RGB result or performance value; those require actual user images and reference masks.
