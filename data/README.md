# Module 4 data

Phase 8 uses a small, real sample from the AAU VAP Trimodal People Segmentation Dataset,
version 3, obtained through the public [Kaggle distribution](https://www.kaggle.com/datasets/aalborguniversity/trimodal-people-segmentation).
The [official AAU project page](https://vap.aau.dk/vap-trimodal-people-segmentation-dataset/)
is the primary provenance reference. The dataset is listed as CC BY 4.0; the academic citation
is recorded in [`experiment_manifest.json`](experiment_manifest.json) and
[`docs/EXPERIMENTAL_RESULTS.md`](../docs/EXPERIMENTAL_RESULTS.md).

The fixed Phase 8 selection is Scene 1 frame IDs `00085`, `00135`, and `00185`, each with RGB,
thermal, and the corresponding dataset person mask. The rule was fixed before metric computation;
it does not use reference masks or performance scores. Exact project-relative paths, ROIs,
source mask values, canonicalization metadata, and per-case OpenCV RNG seeds are in
[`experiment_manifest.json`](experiment_manifest.json).

Frame `00035` may also exist locally as an acquisition reconnaissance file; it is not in the
manifest or Phase 8 result set and was not included in any metric summary.

Acquisition and integrity procedure:

1. Download only the six selected RGB/thermal files and six matching dataset masks from the
   public dataset distribution.
2. Record the source URL, dataset version, license, and source-label values in the manifest.
3. Convert each raw label image to a derived canonical binary mask by mapping zero to background
   and nonzero documented person labels to foreground. This is a representation conversion, not
   fabricated ground truth.
4. Keep all raw and derived input files ignored under `data/rgb/`, `data/thermal/`, and
   `data/reference/`; the Phase 8 manifest contains paths but no machine-specific absolute paths.

The thermal JPGs are false-color three-channel displays, not calibrated temperature arrays. The
thermal pipeline explicitly records that representation and converts BGR display values to an
intensity image. Synthetic arrays in automated tests are fixtures, not experimental evidence.
