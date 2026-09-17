# CSc 8830 Module 4 - Human Boundary Detection and Fourier Theory

This is the independent Module 4 repository for Georgia State University CSc 8830 Computer
Vision. The assignment will provide classical OpenCV human-boundary pipelines for RGB and
thermal images, a strictly separated SAM2 reference comparison, and a Fourier-domain theory
write-up covering Parts A-F.

## Current status

Phase 2 implements the ROI-assisted classical RGB pipeline and its Streamlit page. Thermal
segmentation, SAM2 comparison, metrics, Fourier theory, and empirical results are scheduled for
later approved phases. No results or reference masks are fabricated.

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

The current app exposes four pages:

- RGB Human Boundary (Phase 2 implemented)
- Thermal Human Boundary
- Comparison and Evaluation
- Fourier Theory

Only the RGB page performs processing in this phase. The other pages remain pending-safe.

The standalone app does not require SAM2.

## Run tests

    python -m pytest -q

Tests cover image validation, BGR decoding, canonical boolean masks, metadata, and the
dashboard-compatible page-provider contract. They do not count as experimental validation.

## Planned reproduction workflow

Later phases will add the following commands and outputs:

    python scripts/run_experiments.py
    python scripts/build_report.py

User images will be supplied under the data directories or through the app. Large datasets,
model checkpoints, and user-collected reference masks are not committed by default. A sample
may be bundled only after its provenance and license/terms are verified. Until real images and
reference masks exist, metrics and RGB-versus-thermal observations remain pending user data
collection.

## Architecture

    app.py
    src/module4/
      core CV and theory modules
      reference/          optional SAM2 adapter, isolated from classical processing
      webapp/             PageSpec provider and Streamlit UI
    data/                 user or verified sample inputs
    results/              derived masks, overlays, comparisons, and metrics
    docs/                 methods, theory, results, report, and demo notes
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
- Thermal processing must distinguish intensity data from false-color display data.
- SAM2 is a comparison/reference method only and will remain optional.
- No claim of experimental accuracy, robustness, or RGB-versus-thermal superiority will be made
  before actual user experiments.
