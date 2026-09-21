# CSc 8830 Module 4 - Human Boundary Detection and Fourier Theory

This is the independent Module 4 repository for Georgia State University CSc 8830 Computer
Vision. The assignment will provide classical OpenCV human-boundary pipelines for RGB and
thermal images, a strictly separated SAM2 reference comparison, and a Fourier-domain theory
write-up covering Parts A-F.

## Current status

Phase 2 implements the ROI-assisted classical RGB pipeline, Phase 3 implements the classical
thermal pipeline with dual-polarity Otsu segmentation, Phase 4 implements strict reference
validation and pixel-level evaluation, Phase 5 implements an isolated optional official SAM2
reference adapter plus comparison workflow, Phase 6 implements the Fourier Parts A–F theory and
deterministic educational demonstrations, and Phase 7 completes assignment-wide Streamlit
integration and UX polish. Phase 8 contains a controlled real-data RGB/thermal run on six fixed
AAU VAP cases with generated masks, overlays, and metrics. Official SAM2.1 Hiera Tiny inference
has now completed for the same six cases in an isolated official environment using Apple MPS; the
additional masks, native scores, comparisons, and provenance are stored separately from the
existing dataset-ground-truth results.

## Setup

Python 3.10 or newer is required.

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -U pip
    python -m pip install -e ".[dev]"

On Windows PowerShell, activate with .venv\\Scripts\\Activate.ps1.

## Run the app

From this repository root:

    streamlit run app.py

The current app exposes four consistently ordered pages:

- Question 1 — RGB Human Boundary
- Question 2 — Thermal Human Boundary
- Supporting evaluation — Comparison and Evaluation
- Question 3 — Fourier Theory (Parts A–F)

The RGB and Thermal pages perform classical OpenCV processing and expose their intermediate
sequence. The RGB page requires a user-supplied ROI. The Thermal page evaluates both bright and
dark Otsu hypotheses and distinguishes source intensity data from false-color display palettes.
Comparison and Evaluation runs one classical pipeline, validates an explicitly uploaded reference
or optional SAM2 reference segmentation, and reports metrics only when a valid reference is
available. Fourier Theory visually separates Parts A–F theory from uploaded or deterministic
educational demonstrations.
The canonical written theory is [docs/FOURIER_THEORY.md](docs/FOURIER_THEORY.md).

The standalone app does not require SAM2.

### Optional SAM2 reference environment

The base installation intentionally does not install PyTorch or SAM2. The optional adapter uses
a separate official `facebookresearch/sam2` checkout/environment. Follow the official SAM2
installation instructions, keep the checkpoint in the ignored local `checkpoints/` directory or
outside this repository, and set `MODULE4_SAM2_CHECKPOINT` to its local path before using the SAM2
option in Comparison and Evaluation. The exact API, model/config, checkpoint, prompt provenance,
and completed evidence are documented in [docs/SAM2_COMPARISON.md](docs/SAM2_COMPARISON.md).

## Run tests

    python -m pytest -q

Tests cover image validation, BGR/unchanged decoding, canonical boolean masks, RGB and thermal
pipeline behavior, strict evaluation metrics and alignment, typed experiment records, Fourier
reconstruction/filter/derivative/Laplacian/local-frequency behavior, metadata, and the
dashboard-compatible page-provider contract. They do not count as experimental validation.

## Phase 8 reproduction workflow

An experiment run requires an explicit JSON configuration containing the user's input/reference
paths. The runner never invents rows or metrics:

    python scripts/run_phase8_evidence.py \
      --manifest data/experiment_manifest.json \
      --project-root . \
      --output-dir results

See [docs/EXPERIMENTAL_RESULTS.md](docs/EXPERIMENTAL_RESULTS.md) for provenance, the fixed
selection rule, reference-validation rules, generated metrics, and evidence inventory. The
general-purpose `scripts/run_experiments.py` remains available for new user manifests; a pending
reference produces null metrics and is never represented as a zero score.

Large datasets, model checkpoints, and user-collected reference masks are not committed by
default. The Phase 8 sample provenance and source-label conversion are documented in
[data/README.md](data/README.md) and the manifest.

## Phase 8 official SAM2 evidence

Run the fixed six-case SAM2 comparison from an isolated environment containing the official SAM2
package, PyTorch, TorchVision, and OpenCV:

    PYTHONPATH="$PWD/src" python scripts/run_sam2_phase8_evidence.py \
      --manifest data/experiment_manifest.json \
      --project-root . \
      --output-dir results \
      --checkpoint checkpoints/sam2.1_hiera_tiny.pt \
      --model-name sam2.1_hiera_tiny \
      --model-config configs/sam2.1/sam2.1_hiera_t.yaml \
      --device mps \
      --implementation-version <official-sam2-commit>

The exporter uses the six unchanged manifest cases and supplies each predefined manifest ROI
independently as a SAM2 box. It selects masks only by SAM2-native score and preserves the original
Phase 8 classical-versus-dataset-ground-truth records. See
[docs/EXPERIMENTAL_RESULTS.md](docs/EXPERIMENTAL_RESULTS.md) and
[docs/SAM2_COMPARISON.md](docs/SAM2_COMPARISON.md) for the actual recorded setup and results.

## Final submission packaging

The final Word report, PDF export, demonstration recording, and local submission-preparation
notes are maintained outside version control. The `deliverables/` directory, report builder, demo
notes, and readiness audit are intentionally ignored so the public repository contains the
implementation, reproducibility scripts, technical documentation, tests, and traceable
experiment evidence. The final PDF and demonstration video are submitted separately through the
course workflow.

## Architecture

    app.py
    src/module4/
      core CV, reference adapter, Fourier helpers, and theory modules
      webapp/             PageSpec provider and Streamlit UI
    data/                 user or verified sample inputs
    results/              derived masks, overlays, comparisons, and metrics
    docs/                 methods, theory, results, and technical documentation
    tests/                deterministic automated tests

Core processing will remain importable without Streamlit and will be reused by scripts and the
web app. Module 4 has no runtime dependency on Module 2 or Module 3.

## Optional shared dashboard

When multiple independent module repositories are placed beside one another, a host can mount
Module 4 with:

    from module4.webapp.pages import get_pages

The host should add Module_4/src to its import path and adapt page objects by their
module_label, page_label, order, and render attributes. The course root dashboard already has
this structural compatibility shape. The standalone Module 4 app remains the recommended
grading path.

## Limitations and integrity notes

- Traditional still-image human segmentation is scene-dependent and will document any ROI or
  initialization requirement.
- Thermal processing distinguishes source intensity data from false-color display data, and
  considers both bright and dark foreground polarity.
- SAM2 is an optional reference segmentation, never ground truth. Its adapter is isolated under
  `src/module4/reference/`, has no base dependency, and does not influence either classical
  pipeline. The fixed six-case official SAM2 evidence is recorded separately from the classical
  ground-truth evidence.
- Reference masks are not segmentation inputs. The classical prediction is completed before a
  reference is loaded or evaluated.
- No claim of experimental accuracy, robustness, or RGB-versus-thermal superiority will be made
  before actual user experiments.
