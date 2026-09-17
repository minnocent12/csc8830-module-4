# Module 4 Implementation Plan

## 1. Purpose and planning boundary

This plan covers CSc 8830 Module 4:

1. classical human boundary detection in RGB images;
2. classical human boundary detection in thermal images, with SAM2 used only as a separated comparison/reference method; and
3. Fourier-domain edge detection and region-segmentation theory, Parts A-F.

This is the discovery/planning deliverable requested for the first run. No Module 4 implementation, data collection, SAM2 inference, experiment, Git initialization, commit, or push is authorized by this plan. The first implementation phase requires explicit approval.

Claims about experiment outputs will be classified as pending user data collection until real images, reference masks, and measured results exist.

## 2. Sources of truth

| Source | Role |
| --- | --- |
| AGENTS.md | Root course rules, scientific-integrity rules, Python/OpenCV rules, web-app contract, and scope discipline. |
| GIT_WORKFLOW.md | Git/GitHub protocol. The root AGENTS.md instruction that the agent must not execute Git commands takes precedence for this workspace. |
| Module_4/AGENTS.md | Module 4-specific constraints, including classical-only pipelines, SAM2 isolation, evaluation, and web-app surface. |
| Module_4/MODULE_4_ASSIGNMENT_INSTRUCTIONS.md | Submission requirements for repository, code, web app, video, PDF, and digitized hand-worked material. |
| Module_4/MODULE_4_ASSIGNMENT_QUESTIONS.md | Authoritative Question 1, Question 2, Question 3 Parts A-F, evaluation criteria, and required deliverables. |
| User-pasted implementation brief | Planning-only gate, required first-run contents, architecture constraints, and phase-gate procedure. |

## 3. Requirements inventory

### Question 1: RGB human boundary detection

- Accept a standard RGB/color image.
- Use Python, OpenCV, and NumPy; use Matplotlib where visualization is useful.
- Keep the required segmentation entirely classical: color/intensity processing, thresholding, morphology, connected components, contours, region methods, or GrabCut.
- Do not use a learned detector, classifier, neural network, foundation model, or hidden ML dependency.
- Produce preprocessing and meaningful intermediate results.
- Isolate the human region as a canonical binary mask.
- Extract the human contour/boundary separately from the mask.
- Produce a final boundary overlay on the unchanged original image.
- Provide a SAM2 reference result only as an independent comparison.
- Support visual comparison and numerical evaluation where a reference mask exists.

### Question 2: thermal human boundary detection

- Accept true grayscale/intensity thermal data and document handling of false-color thermal inputs.
- Normalize safely while preserving the distinction between source data and display data.
- Use a fully classical pipeline with explicit foreground polarity handling.
- Produce preprocessing, threshold, binary-mask, contour, and final-boundary outputs.
- Provide an independent SAM2 reference comparison without allowing SAM2 to influence the classical pipeline.
- Evaluate visually and numerically where a reference or ground-truth mask exists.
- Discuss RGB versus thermal only from actually observed tested images.

### Question 3: Fourier theory, Parts A-F

- Part A: continuous and discrete 2D Fourier representation, inverse transform, low/high spatial frequencies, magnitude, and phase.
- Part B: mathematical relationship between discontinuities/edges and high frequencies.
- Part C: high-pass filtering, including H_HP(u,v) = 1 - H_LP(u,v), multiplication, and inverse transform.
- Part D: derivative property in x and y and why frequency multiplication emphasizes edges.
- Part E: spatial Laplacian and its frequency-domain transfer function -4*pi^2*(u^2+v^2).
- Part F: frequency-selective region segmentation, global versus local frequency analysis, windowed transforms, advantages, limitations, boundary/window effects, and computation.

### Submission and evidence requirements

- Maintain a separate accessible GitHub repository for Module 4.
- Document dependencies, setup, execution, methods, and reproducibility in a README.
- Expose every assignment component through a working web application.
- Prepare a demonstration-video checklist; the user must record the video.
- Prepare report-ready figures, tables, theory, and a final PDF workflow.
- Digitize any required solve-by-hand/show-by-example work as typed or scanned content; do not use ordinary photographs of paper.

## 4. Workspace and repository findings

Filesystem inspection found:

- Module_2/.git exists.
- Module_3/.git exists.
- Module_4/.git does not exist.
- The Assignments root has no .git directory.
- Module 4 currently contains AGENTS.md, CLAUDE.md, and the two authoritative assignment documents only.
- The root app.py already lists module4.webapp.pages as a future provider and skips the module when it is not importable.
- Module 2 and Module 3 each have independent app.py, pyproject.toml, src/, tests/, README.md, data, results, and documentation structures.
- No Module 4 image dataset, reference mask, result, or experiment value was found.

Live Git branch/status was not queried because the root AGENTS.md explicitly prohibits the agent from executing Git commands, including read-only commands. Before implementation, the user should run and report the output of these commands from the Assignments root:

    git -C Module_2 status --short --branch
    git -C Module_3 status --short --branch
    git -C Module_4 status --short --branch
    git -C Module_4 rev-parse --show-toplevel
    git -C Module_3 remote -v
    git -C Module_2 remote -v

The Module 4 commands are expected to fail until its separate repository is created. No repository boundary will be initialized, moved, or changed during this planning pass.

## 5. Reusable patterns from earlier modules

### Module 2 patterns worth reusing

- Standalone thin app.py that adds src/ for direct execution and delegates to the page provider, registry, and shell.
- Independent package under src/module2, with core processing free of Streamlit imports.
- Dataclass-style page contract with module_label, page_label, order, and render.
- Provider-agnostic page registry with validation, de-duplication, and stable ordering.
- Explicit pending-experiment banners and documentation of user-supplied empirical data.
- pyproject.toml package discovery, Python version constraint, runtime dependencies, and a pytest development extra.
- Deterministic unit tests for mathematical and validation functions.
- README sections for setup, app execution, experiments, limitations, and standalone versus course-dashboard use.

### Module 3 patterns worth reusing

- The clean four-page Streamlit organization: primary operation, comparison/validation, experimental results, and theory.
- Small webapp/_page.py, registry.py, shell.py, ui.py, and pages.py files matching the shared dashboard contract.
- Reuse of core functions by both Streamlit pages and experiment scripts.
- Explicit BGR-to-RGB conversion at display boundaries and clear grayscale/float handling.
- Generated results in results/ and report figures in docs/report/figures/.
- Synthetic deterministic fixtures for tests, separate from academic experimental data.
- Page-provider contract tests compatible with the root dashboard structural adapter.

### Patterns not to copy

- Do not import Module 2 or Module 3 code at runtime. Module 4 must be independently installable and submittable.
- Do not copy Module 3's blurring-specific core or its numerical experiment assumptions.
- Do not treat a deterministic synthetic fixture as a real RGB/thermal experiment.
- Do not pre-populate empirical IoU, Dice, precision, recall, timing, or SAM2 values.
- Do not use a root dashboard as a Module 4 dependency or modify it during the initial implementation.
- Do not put CV processing inside Streamlit callbacks.
- Do not represent a contour as though it were the segmentation mask.

## 6. Proposed Module 4 file tree

This is the proposed target structure, subject to adjustment only if implementation findings require it:

    Module_4/
    ├── AGENTS.md
    ├── CLAUDE.md
    ├── IMPLEMENTATION_PLAN.md
    ├── MODULE_4_ASSIGNMENT_INSTRUCTIONS.md
    ├── MODULE_4_ASSIGNMENT_QUESTIONS.md
    ├── README.md
    ├── app.py
    ├── pyproject.toml
    ├── .gitignore
    ├── data/
    │   ├── rgb/.gitkeep
    │   ├── thermal/.gitkeep
    │   ├── reference/.gitkeep
    │   └── README.md
    ├── docs/
    │   ├── RGB_PIPELINE.md
    │   ├── THERMAL_PIPELINE.md
    │   ├── FOURIER_THEORY.md
    │   ├── DATA_AND_PROVENANCE.md
    │   ├── SAM2_COMPARISON.md
    │   ├── EXPERIMENTAL_RESULTS.md
    │   ├── DEMO_VIDEO_CHECKLIST.md
    │   ├── REPORT_NOTES.md
    │   └── report/figures/.gitkeep
    ├── results/
    │   ├── rgb/.gitkeep
    │   ├── thermal/.gitkeep
    │   ├── comparisons/.gitkeep
    │   ├── metrics/.gitkeep
    │   └── README.md
    ├── scripts/
    │   ├── run_experiments.py
    │   └── build_report.py
    ├── src/module4/
    │   ├── __init__.py
    │   ├── types.py
    │   ├── io_utils.py
    │   ├── preprocessing.py
    │   ├── components.py
    │   ├── contours.py
    │   ├── rgb.py
    │   ├── thermal.py
    │   ├── metrics.py
    │   ├── validation.py
    │   ├── visualization.py
    │   ├── fourier.py
    │   ├── reference/
    │   │   ├── __init__.py
    │   │   └── sam2_reference.py
    │   └── webapp/
    │       ├── __init__.py
    │       ├── _page.py
    │       ├── registry.py
    │       ├── shell.py
    │       ├── ui.py
    │       └── pages.py
    └── tests/
        ├── test_io_utils.py
        ├── test_components.py
        ├── test_contours.py
        ├── test_rgb.py
        ├── test_thermal.py
        ├── test_metrics.py
        ├── test_fourier.py
        ├── test_reference.py
        └── test_page_contract.py

The package may combine components.py and contours.py if the resulting code is clearer; the goal is cohesive, testable responsibilities, not maximum file count.

## 7. Core architecture

types.py will hold explicit dataclasses/enums for pipeline configuration and structured results. A result will preserve the original shape and include named stages, a canonical mask, contours, boundary overlay, warnings, and provenance metadata.

io_utils.py will validate paths/bytes, decode BGR color input, load unchanged thermal data, and enforce dtype/shape rules. Original arrays will never be mutated.

preprocessing.py will contain documented color/intensity normalization and morphology helpers. components.py will contain classical connected-component selection. contours.py will extract and represent external contours separately from masks.

rgb.py and thermal.py will contain the inspectable segmentation pipelines. metrics.py will contain mask normalization and IoU/Dice/precision/recall. validation.py will validate reference alignment and assemble result rows. visualization.py will render display-only uint8 images, overlays, spectra, and figures.

fourier.py will contain only NumPy/OpenCV-compatible frequency operations for theory demonstrations. reference/sam2_reference.py will be optional and isolated. The webapp package will contain UI dispatch only.

Canonical internal mask convention:

    numpy bool array
    False = background
    True  = human foreground

Exported/displayed masks will be uint8 with values 0 and 255. Metrics will require equal height, width, orientation, and foreground convention; any explicitly necessary nearest-neighbor mask resize will be documented and surfaced in metadata.

## 8. Proposed RGB strategy

### Primary pipeline

The primary reproducible approach will be a classical, ROI-assisted GrabCut pipeline:

1. Load a BGR uint8 image without modifying it.
2. Validate dimensions/channels and expose BGR versus RGB explicitly.
3. Produce useful diagnostics: RGB display, grayscale, and a Lab/HSV representation for inspection of color separation and lighting.
4. Accept a user-supplied rectangle around the person, because a traditional method cannot reliably identify a human in arbitrary clutter without either an initialization or strong scene assumptions. The UI and report will call this ROI-assisted, not fully automatic.
5. Run cv2.grabCut with GC_INIT_WITH_RECT on the color image. No detector or learned model will choose the rectangle or alter the result.
6. Convert GrabCut labels to the canonical boolean mask.
7. Apply documented opening/closing parameters to remove small noise and close small gaps.
8. Use connected components and ROI overlap/area rules to retain the intended foreground.
9. Extract external contour(s), preserving the mask and contour as separate outputs.
10. Draw the contour on a copy of the original BGR image and convert only at display time.

The pipeline will expose its initialization, raw GrabCut mask, cleaned mask, selected component, final mask, contour, and boundary overlay. If the input has no suitable ROI or the result is ambiguous, the result will contain a warning rather than silently claiming a human was found.

### Alternatives considered

- HSV/Lab thresholding: simple and explainable, but skin/clothing/background colors vary and it is not reliable as the primary method for arbitrary RGB scenes. It remains useful as a diagnostic or optional threshold mode.
- Canny alone: produces edges rather than a closed human region and is sensitive to texture, lighting, and broken boundaries.
- Background subtraction: useful for controlled video/camera scenes, but not appropriate for one arbitrary still image without a background model.
- Watershed/region growing: possible classical alternatives, but they need seeds or strong intensity assumptions and add failure modes without improving the base contract.
- Automatic saliency/person detectors: excluded because they are not guaranteed to remain classical and could hide learned behavior.

## 9. Proposed thermal strategy

The thermal pipeline will support one-channel intensity images and false-color three-channel inputs. True source data will be loaded with unchanged dtype where possible. Processing will normalize finite intensity values to uint8 for OpenCV operations while retaining source dtype and normalization metadata.

Pipeline:

1. Validate the thermal image and determine grayscale versus false-color input.
2. Convert false-color input to grayscale explicitly, with a warning that display color is not necessarily physical temperature.
3. Apply only documented preprocessing, such as optional Gaussian denoising or CLAHE, and record all parameters.
4. Compute Otsu candidates for both bright-foreground and dark-foreground polarity. Do not assume that humans are always hotter than the background.
5. Apply opening/closing and connected-component filtering to each candidate.
6. If an ROI is supplied, use classical overlap/area rules to select the candidate component. Without an ROI, score candidates using transparent geometry/area/centrality heuristics and return an ambiguity warning when the score is not decisive.
7. Produce normalized intensity, threshold candidates, cleaned mask, selected component, final mask, contour, and boundary overlay.

This keeps polarity selection classical and inspectable. It does not make a universal claim that thermal imagery is easier or that the brightest component is always the person.

### Thermal alternatives considered

- Adaptive thresholding: useful when illumination/intensity varies spatially, but can break a person into pieces in low-resolution thermal data; it will be an optional comparison mode.
- Fixed percentile thresholds: useful for controlled sensor ranges, but not portable across sensors and temperature scenes.
- Region growing: possible with a user seed/ROI, but more sensitive to intensity gradients and less reproducible than Otsu plus morphology for the initial implementation.
- Thermal background subtraction: useful for video sequences, not justified for isolated stills unless the user supplies a background model.

## 10. Contour and boundary design

The final result will distinguish:

- mask: filled boolean human region;
- contours: OpenCV point arrays extracted from the selected mask; and
- boundary_overlay: a copy of the original image with the contour drawn.

The default contour policy will use external contours. A largest/ROI-overlapping component will be selected when the task expects one human; multiple external contours can be preserved in diagnostics when the mask contains disconnected foreground. The report will state the selection policy, contour retrieval mode, thickness, and colors.

## 11. Intermediate-result design

Each stage must explain an algorithmic purpose and be available to the script and web app:

| Modality | Stages |
| --- | --- |
| RGB | original BGR, grayscale/color-space diagnostic, ROI initialization, raw GrabCut labels, cleaned candidate, selected component, final mask, contour, boundary overlay |
| Thermal | original unchanged source, normalized intensity, optional enhancement, bright/dark threshold candidates, cleaned candidate, selected component, final mask, contour, boundary overlay |

Display conversions will be separate from computational arrays. No stage will overwrite the original input.

## 12. SAM2 isolation and comparison

reference/sam2_reference.py will expose a small adapter interface with explicit statuses such as available, completed, pending, and failed. It will use lazy optional imports so the classical pipelines and the base app work when SAM2, PyTorch, weights, or GPU support are not installed.

The adapter will accept an image and a separately supplied prompt/configuration, and return a reference mask plus provenance. It will not be imported by rgb.py or thermal.py, and it will not choose an ROI, threshold, morphology setting, component, contour, or repair step.

Reference-mask validation will check dimensions, orientation, binary convention, and image identity. Reference masks will be aligned only when the transformation is known and recorded; nearest-neighbor interpolation is required for any documented binary resize. If SAM2 has not actually completed, the UI, CSV/JSON, and report will say PENDING USER EXPERIMENT and contain no invented masks or metrics.

Agreement with SAM2 will be described as agreement with a reference method, not proof of objective correctness. If a user supplies ground truth, both OpenCV-versus-ground-truth and SAM2-versus-ground-truth evaluations will be separate.

## 13. Evaluation methodology

For each aligned pair, let A be the classical OpenCV boolean mask and B be the explicit reference mask. Implement and test:

    intersection = |A AND B|
    union        = |A OR B|
    IoU          = intersection / union
    Dice         = 2 * intersection / (|A| + |B|)
    precision    = TP / (TP + FP)
    recall       = TP / (TP + FN)

Edge cases will be explicit: perfect agreement of two empty masks returns a documented convention, one empty mask versus a non-empty mask returns zero, and shape mismatch raises a validation error. Metrics are not computed when reference status is pending.

The experiment table will include image ID, modality, method, reference source, metrics, status, and notes. Visual evidence will include the classical mask, reference mask, boundary overlay, and difference/overlap visualization. Optional boundary-distance metrics may be added only if their coordinate and tolerance definitions are clear.

## 14. Dataset and sample-data strategy

No Module 4 sample data is currently present. Phase 1 will create ignored data directories and document an input manifest rather than invent provenance.

The target strategy is:

- add a small, legally redistributable RGB sample only after source, license, checksum, and preprocessing are verified;
- add a small public thermal sample only after source URL, license/terms, modality, dimensions, and download/reproduction instructions are verified;
- keep larger datasets and user images outside the repository;
- support uploads for all pages;
- use deterministic synthetic scenes only for automated tests, clearly labeled as fixtures;
- provide a data-entry/reference-mask directory for user-collected masks.

If a legal, reproducible bundled sample cannot be verified, the app will remain upload-capable and report sample-dependent results as pending. No dataset provenance or experimental outcome will be guessed.

## 15. Experiment matrix

The planned runner will be deterministic for code-controlled settings and will write only results produced by actual execution:

| Case | Input | Classical operation | Reference/evaluation | Artifacts |
| --- | --- | --- | --- | --- |
| RGB-1..n | verified RGB samples or user images | ROI-assisted GrabCut plus cleanup | SAM2/reference or ground truth if actually available | all stages, mask, contour, overlay, metrics/status |
| TH-1..n | verified thermal samples or user images | dual-polarity Otsu plus cleanup | SAM2/reference or ground truth if actually available | all stages, mask, contour, overlay, metrics/status |
| Theory demos | deterministic grayscale fixture | FFT spectrum, low/high-pass, derivative, Laplacian, local windows | numerical reconstruction/sanity checks | report-ready figures and metadata |

run_experiments.py will fail clearly for missing required inputs or write explicit pending rows where the workflow is designed to continue without a reference. It will not type metrics into a document manually.

## 16. Fourier theory plan for Parts A-F

docs/FOURIER_THEORY.md will be the canonical report-ready theory document:

- Part A: define f(x,y), the continuous transform, the discrete 2D DFT, inverse transform, complex coefficients, magnitude |F(u,v)|, phase arg(F(u,v)), and low/high spatial frequencies.
- Part B: connect step changes and rapid spatial variation to high-frequency spectral content; define filtering as G = H F and reconstruction as g = F^-1{G}.
- Part C: derive H_HP = 1 - H_LP, apply it to F, and explain the resulting edge emphasis.
- Part D: derive/use F{df/dx} = j2*pi*u F(u,v) and the y counterpart, including the frequency-amplification interpretation.
- Part E: derive the second-derivative terms and F{nabla^2 f} = -4*pi^2*(u^2+v^2) F(u,v); connect the transfer function to edge enhancement and noise sensitivity.
- Part F: describe frequency-selective region segmentation using filtered reconstruction, frequency-feature thresholds, local/windowed transforms f_k and F_k, texture-region separation, spatial-localization loss, window/boundary effects, computational cost, advantages, and limitations without ML.

Any notation used in the document will define variables, coordinate conventions, normalization, and discrete boundary assumptions. The equations will match the optional demonstration code.

## 17. Optional theory visualizations

If useful after the core theory is complete, fourier.py and the Theory page will show:

- original grayscale fixture;
- centered log-magnitude spectrum;
- low-pass and high-pass reconstructions;
- derivative and Laplacian frequency weights;
- local-window spectral energy/texture visualization.

These visuals support, but do not replace, the Parts A-F derivation. They will be labeled as illustrations unless generated from a real experiment.

## 18. Streamlit/page architecture

Module 4 will expose a provider through module4.webapp.pages.get_pages() and define its own PageSpec-compatible dataclass, registry, shell, and UI helpers, following Module 3 without creating a runtime dependency on it.

Proposed pages:

1. RGB Human Boundary: upload/select image, enter ROI, run classical OpenCV pipeline, inspect all stages, contour, overlay, optional SAM2 reference, and metrics/status.
2. Thermal Human Boundary: upload/select thermal image, choose normalization/options and polarity/ROI controls, inspect all stages, contour, overlay, optional reference, and metrics/status.
3. Comparison and Evaluation: select completed artifacts/reference masks, show overlap and metrics tables, download CSV/JSON, and explain reference-versus-ground-truth semantics.
4. Fourier Theory: render Parts A-F from the canonical document and, where available, show supporting frequency-domain demonstrations.

The standalone Module_4/app.py will collect Module 4 pages. Core modules will not import Streamlit. The root dashboard will be left unchanged; its existing structural adapter can mount Module 4 when module4.webapp.pages becomes importable.

## 19. Testing strategy

Tests will run without a GPU, SAM2, or user data:

- common: image decoding/validation, dtype/range handling, mask normalization, shape checks, contour extraction, component selection, and immutability of inputs;
- RGB: synthetic foreground/background scenes, ROI validation, GrabCut output shape and binary convention, morphology and contour smoke tests;
- thermal: synthetic hot-on-cool and cool-on-hot scenes, Otsu threshold candidates, polarity selection, morphology, component filtering, and false-color conversion;
- metrics: tiny hand-verifiable masks for perfect overlap, no overlap, partial overlap, false positives, false negatives, both empty, and shape mismatch;
- Fourier: FFT shape, inverse reconstruction, centered spectrum, low/high-pass masks, real output, derivative/Laplacian weighting, and local-window output shape;
- reference adapter: pending/failed interface and mask normalization only. No test will claim successful SAM2 inference without a real run;
- webapp: page-provider type/order/merge contract matching the Module 2/3 pattern.

Automated tests will be labeled tested; they will not be presented as experimental validation.

## 20. Documentation strategy

The README will cover purpose, features, structure, Python version, dependencies, setup, standalone app execution, optional shared-dashboard mounting, experiment commands, tests, input expectations, output paths, SAM2 setup/status, provenance, reproducibility, and limitations.

The method documents will explain color spaces, dtypes, thresholds, kernels, morphology, component rules, contour semantics, ROI assumptions, polarity rules, and mask alignment.

Results documents will contain generated tables and figures only after execution, with pending sections where user data or SAM2 is absent. A report script will assemble Markdown and optional PDF/DOCX outputs without copying source listings into the final report.

The final report will include the assignment mapping, methods, intermediate/final figures, reference comparison, actual metrics, RGB-versus-thermal observations, Parts A-F, limitations, repository link, and conclusion. The demo checklist will mirror the required app flow.

## 21. Git and repository strategy

Module 4 is intended to become a separate repository rooted at Assignments/Module_4, matching Module 2 and Module 3. It will not be nested as a runtime dependency of the other modules and will not alter their repositories.

During this planning pass, no .git directory will be created, no repository will be moved or deleted, and no Git command will be executed by the agent. After the user confirms the repository strategy and runs the required Git inspection, the user must establish the separate repository and a normal task branch before implementation. The branch must not be main or master; a suitable first branch is task/module-4-foundation.

No commit, push, pull request, merge, or history rewrite is part of this plan. The user remains the Git/GitHub decision maker and must ensure instructor access and the final repository link.

## 22. Reproducibility strategy

- Pin minimum/reproducible dependency versions in pyproject.toml, with a development extra for pytest.
- Use pathlib, typed structured results, explicit seeds where randomness is used, and no mutation of original inputs.
- Record input filename, checksum, dimensions, dtype, processing parameters, ROI, threshold, polarity, mask convention, and reference provenance in result metadata.
- Reuse exactly the same core pipeline from scripts and Streamlit.
- Save intermediate outputs to new files under results; keep source/user data separate.
- Use nearest-neighbor only for documented binary-mask alignment.
- Keep SAM2 optional and record model/checkpoint/device details only when a real inference succeeds.
- Mark unavailable samples, references, metrics, and empirical discussion as pending rather than filling plausible values.

## 23. Phased execution roadmap

Every phase is gated. The next phase starts only after the user explicitly says Proceed with Phase N. Each approved phase will include a fresh user-reported Git status, scope restatement, only that phase's edits, relevant tests, diff review, changed-file list, exact test results, skipped tests/reasons, deviations, acceptance decision, and recommendation.

### Phase 0: discovery and architecture

- Goal: complete the requirements checklist, repository assessment, and this plan.
- Scope/files: Module_4/IMPLEMENTATION_PLAN.md only; no code or data.
- Tests/checks: filesystem inspection and document review; no implementation test.
- Acceptance: all required plan sections and traceability rows exist; no empirical claims are presented as results.
- Artifacts: this plan.
- Dependencies/risks: Git live state remains user-supplied because of the root prohibition.
- Out of scope: Module 4 code, repository initialization, datasets, SAM2, app changes.

### Phase 1: foundation and data interfaces

- Goal: create the independent Python package skeleton, IO/mask types, pending-data layout, configuration, and page-provider skeleton.
- Files: pyproject.toml, .gitignore, app.py, src/module4/{__init__,types,io_utils}.py, src/module4/webapp/*, data/results/docs directory placeholders, initial README, and foundational tests.
- Tests: IO validation, mask convention, immutability, page contract, package/import smoke tests.
- Acceptance: clean install/import, standalone app starts to pending notices, tests pass, no other module is imported, and no empirical output is generated.
- Artifacts: independently runnable skeleton and pending-data instructions.
- Dependencies/risks: user must establish the Module 4 repository/branch; bundled sample provenance is still unresolved.
- Out of scope: segmentation algorithms, SAM2 execution, final report, and experiments.

### Phase 2: RGB classical pipeline

- Goal: implement and test the ROI-assisted classical RGB boundary pipeline.
- Files: preprocessing.py, components.py, contours.py, rgb.py, visualization updates, RGB tests, and RGB method documentation.
- Tests: synthetic scene tests, ROI/input errors, output-shape/binary checks, and contour overlay checks.
- Acceptance: deterministic classical output, transparent ROI requirement, preserved stages, separate mask/contour, no ML/DL imports, and no original-image mutation.
- Artifacts: RGB pipeline callable from Python and ready for app integration.
- Dependencies/risks: GrabCut quality depends on ROI and scene contrast; no universal automatic-human claim will be made.
- Out of scope: SAM2, final metrics, and empirical claims.

### Phase 3: thermal classical pipeline

- Goal: implement and test thermal normalization, dual-polarity thresholding, component selection, contour extraction, and boundary visualization.
- Files: thermal.py, related preprocessing/components/visualization updates, thermal tests, and thermal method documentation.
- Tests: hot/cool synthetic fixtures, false-color handling, polarity, dtype/range, and ambiguity behavior.
- Acceptance: explicit foreground polarity, documented parameters, deterministic output on supported inputs, and no ML/DL path.
- Artifacts: thermal pipeline callable from Python and ready for app integration.
- Dependencies/risks: intensity-only thermal data and sensor-specific contrast may limit threshold quality.
- Out of scope: reference inference and user-data claims.

### Phase 4: metrics and reproducible experiment runner

- Goal: implement mask comparison, validation, artifact writing, and experiment orchestration.
- Files: metrics.py, validation.py, scripts/run_experiments.py, metrics tests, and result schemas/templates.
- Tests: exact hand-verifiable metric cases, invalid alignment, pending reference behavior, and script smoke tests on fixtures.
- Acceptance: metrics are correct and documented; result files distinguish tested synthetic fixtures from pending real experiments; scripts reuse core pipelines.
- Artifacts: empty/pending-safe CSV/JSON schema and report-ready artifact layout.
- Dependencies/risks: no metric row may be created for a missing reference.
- Out of scope: SAM2 model execution and final empirical discussion.

### Phase 5: SAM2 reference adapter and comparison

- Goal: add the optional, strictly isolated SAM2/reference workflow.
- Files: reference/sam2_reference.py, adapter tests/docs, comparison UI/script integration, and SAM2_COMPARISON.md.
- Tests: interface/pending/failure/mask-validation tests; integration tests only when a real user-approved model/checkpoint run is available.
- Acceptance: classical output is unchanged when SAM2 is absent; successful reference runs include provenance; unavailable runs remain PENDING USER EXPERIMENT; no fabricated mask or metric exists.
- Artifacts: independently stored reference masks and comparison records when real runs exist.
- Dependencies/risks: SAM2 package, weights, hardware, licensing, and runtime may be absent.
- Out of scope: treating SAM2 as ground truth or primary segmentation.

### Phase 6: Fourier theory and optional demonstrations

- Goal: complete mathematically correct Parts A-F and optional supporting frequency demos.
- Files: docs/FOURIER_THEORY.md, fourier.py, Fourier tests, figures, and theory-page data.
- Tests: transform reconstruction, filter masks, derivative/Laplacian sanity, and local-window shapes/numerical checks.
- Acceptance: every Part A-F requirement is addressed; notation matches code; visuals are labeled as demonstrations and never replace derivations.
- Artifacts: canonical theory document and optional figures.
- Dependencies/risks: FFT normalization, frequency ordering, windowing, and boundary effects must be explained accurately.
- Out of scope: ML-based texture segmentation.

### Phase 7: Streamlit/web application completion

- Goal: expose all assignment components through separate RGB, thermal, comparison, and theory pages.
- Files: app.py, src/module4/webapp/*, UI tests/docs, and README run instructions.
- Tests: provider contract, page import, pending-data behavior, and manual local smoke test with upload/fixture inputs.
- Acceptance: standalone app works without SAM2; all required stages and metrics/status are visible; classical and reference outputs are clearly separated; root dashboard can import the provider without root changes.
- Artifacts: demonstrable web app.
- Dependencies/risks: Streamlit widget state and large image rendering need bounded controls.
- Out of scope: recording the final video.

### Phase 8: real experiments, results, and RGB-versus-thermal analysis

- Goal: execute only the approved, real image/reference workflow and generate evidence.
- Files: data supplied by the user, results/, report figures, generated CSV/JSON/Markdown, and EXPERIMENTAL_RESULTS.md.
- Tests/checks: script rerun, artifact inspection, mask alignment, and consistency checks.
- Acceptance: every reported metric has a real input/reference provenance; observed RGB versus thermal discussion is tied to tested images; missing data remains pending.
- Artifacts: final masks, contours, overlays, comparison figures, and results table.
- Dependencies/risks: user must supply legal images, ROI choices, ground truth/reference masks, and any SAM2 execution.
- Out of scope: inventing values, silently correcting masks, or claiming general robustness.

### Phase 9: documentation, report evidence, and demo preparation

- Goal: synchronize README/docs, assemble report-ready material, and prepare the video script.
- Files: all docs, scripts/build_report.py, REPORT_NOTES.md, DEMO_VIDEO_CHECKLIST.md, and generated report artifacts.
- Tests/checks: documentation link/path checks, report build smoke test, figure presence, and pending-status audit.
- Acceptance: final PDF source includes all assignment sections, actual figures/tables where available, repository link placeholder or confirmed link, limitations, and no fabricated content; video checklist covers app, inputs, processing, outputs, and theory.
- Artifacts: report source/optional PDF, figure manifest, and demo checklist.
- Dependencies/risks: PDF tooling and repository URL may be user/environment dependent.
- Out of scope: recording/uploading video or Google Classroom submission.

### Phase 10: regression and submission readiness

- Goal: perform clean-room reproducibility and final completeness review.
- Files: only fixes required by verified failures; final checklist.
- Tests/checks: fresh environment install, full pytest, app startup, experiment rerun where data exists, report build, file/link audit, and final Git status review by the user.
- Acceptance: implementation/tests/docs/evidence statuses are honest and complete; pending user work is enumerated; separate repository is instructor-accessible; app and report are ready for user video/PDF/Classroom submission.
- Artifacts: final submission package and completion checklist.
- Dependencies/risks: final user data, SAM2 evidence, repository access, recording, and upload.
- Out of scope: Git merge/history rewriting and external submission without user action.

## 24. Phase acceptance summary

No phase after Phase 0 has started. Phase completion means its tests and acceptance criteria actually pass; code existing by itself is not sufficient. Every later report will explicitly separate implemented, tested, experimentally validated, theoretically derived, and pending user data collection claims.

## 25. Risks and real dependencies

- Traditional still-image human detection is scene-dependent. ROI-assisted GrabCut is the planned transparent compromise; the UI must not call it fully automatic.
- Thermal images may be grayscale, 16-bit, or false-color and may not encode calibrated temperature. Normalization and provenance must be recorded.
- Otsu polarity selection can be ambiguous when several objects have similar intensity. The pipeline must warn instead of silently asserting a human region.
- SAM2 requires separate packages/checkpoints/hardware and may not be available. It is optional at runtime and pending until actually executed.
- Reference-mask alignment and image identity are prerequisites for numerical comparison.
- Public sample data requires license/provenance verification before bundling.
- The required final video, GitHub access, PDF upload, and Google Classroom submission remain user-controlled external actions.
- The root dashboard is convenience infrastructure only; Module 4 must remain standalone.

## 26. Definition of done

Module 4 is complete only when all applicable items below are true:

- classical RGB and thermal pipelines are implemented, documented, tested, and reachable from the standalone app;
- masks, contours, intermediate stages, and final boundary overlays are separately represented;
- SAM2 comparison is isolated, provenance-bearing, and absent/pending when not actually run;
- IoU, Dice, precision, and recall are correct, aligned, and traceable to real references;
- Parts A-F are mathematically complete and consistent with optional code demonstrations;
- actual experiments, figures, and RGB-versus-thermal observations are generated or explicitly marked pending;
- README, setup, experiment, test, limitation, report, and demo instructions are synchronized;
- the separate GitHub repository is accessible to the instructor and linked in the PDF;
- the user has recorded the demo video and submitted the final PDF/video through Google Classroom.

## 27. Assignment traceability matrix

| Requirement | Source | Implementation | Verification/Evidence | Phase | Status |
| --- | --- | --- | --- | --- | --- |
| RGB color input | Questions Q1 Requirements | io_utils.py, RGB page | Upload/fixture decode test and app evidence | 1, 2, 7 | planned |
| Classical RGB human isolation | Questions Q1 Requirements; Module 4 AGENTS | rgb.py ROI-assisted GrabCut pipeline | Synthetic tests, saved stages, method doc | 2 | planned |
| RGB preprocessing/intermediate outputs | Questions Q1 Expected Output; brief | preprocessing.py, structured RGB result | Stage files and app screenshots | 2, 7, 8 | planned |
| RGB binary human region | Questions Q1 Requirements | rgb.py, canonical bool mask | Mask convention tests and saved mask | 2, 4, 8 | planned |
| RGB contour/boundary | Questions Q1 Requirements/Expected Output | contours.py, visualization.py | Contour tests and final overlay | 2, 7, 8 | planned |
| RGB SAM2 comparison | Questions Q1; SAM2 rule | reference/sam2_reference.py | Real reference provenance or pending banner | 5, 8 | planned |
| RGB visual/numerical evaluation | Questions comparison/evaluation | metrics.py, comparison page | Overlap figure and metrics row when reference exists | 4, 5, 7, 8 | planned |
| Thermal input support | Questions Q2 Requirements | io_utils.py, thermal.py | Grayscale/false-color/dtype tests | 1, 3 | planned |
| Thermal preprocessing/normalization | Questions Q2 traditional processing | preprocessing.py, thermal.py | Recorded parameters and stage outputs | 3, 8 | planned |
| Thermal classical human isolation | Questions Q2 Requirements | Dual-polarity Otsu/morphology/components | Synthetic hot/cool tests and artifacts | 3 | planned |
| Thermal binary region | Questions Q2 Expected Output | thermal.py canonical bool mask | Mask convention and polarity tests | 3, 4 | planned |
| Thermal contour/boundary | Questions Q2 Requirements/Expected Output | contours.py, visualization.py | Contour tests and final overlay | 3, 7, 8 | planned |
| Thermal SAM2 comparison | Questions Q2; SAM2 rule | Isolated reference adapter | Real reference provenance or pending status | 5, 8 | planned |
| Thermal visual/numerical evaluation | Questions evaluation | metrics.py, comparison page | Metrics/overlap artifacts when reference exists | 4, 5, 7, 8 | planned |
| IoU | Questions Evaluation and Validation | metrics.py | Hand-verifiable exact tests and result CSV | 4, 8 | planned |
| Dice | Questions Evaluation and Validation | metrics.py | Hand-verifiable exact tests and result CSV | 4, 8 | planned |
| Precision | Questions Evaluation and Validation | metrics.py | TP/FP exact tests and result CSV | 4, 8 | planned |
| Recall | Questions Evaluation and Validation | metrics.py | TP/FN exact tests and result CSV | 4, 8 | planned |
| Empty/mismatched mask handling | Module 4 AGENTS/brief | metrics.py, validation.py | Edge-case tests and explicit status/errors | 4 | planned |
| Fourier Part A | Questions Q3 Part A | docs/FOURIER_THEORY.md, fourier.py | Theory review and FFT reconstruction test | 6 | planned |
| Fourier Part B | Questions Q3 Part B | Theory document and spectrum demo | Derivation plus high-frequency illustration | 6 | planned |
| Fourier Part C | Questions Q3 Part C | High-pass helper and theory document | H_HP = 1 - H_LP test/figure | 6 | planned |
| Fourier Part D | Questions Q3 Part D | Derivative-frequency helper and derivation | Numerical sanity check and explanation | 6 | planned |
| Fourier Part E | Questions Q3 Part E | Laplacian-frequency helper and derivation | Transfer-function test/figure | 6 | planned |
| Fourier Part F | Questions Q3 Part F | Region/local-window theory and helpers | Local-window test and theory page | 6, 7 | planned |
| Fourier advantages/limitations | Questions Q3 theory discussion | FOURIER_THEORY.md | Report section covering localization/window/cost | 6, 9 | planned |
| RGB-versus-thermal discussion | Questions RGB vs Thermal Discussion | EXPERIMENTAL_RESULTS.md, report | Tied to actual tested images or pending note | 8, 9 | planned |
| Working web application | Instructions Section 3; Questions web app | app.py, src/module4/webapp | Standalone app smoke test and video | 1, 7, 10 | planned |
| Separate RGB web surface | Module 4 AGENTS/brief | RGB page | App screenshot/video showing inputs/stages/results | 7, 9 | planned |
| Separate thermal web surface | Module 4 AGENTS/brief | Thermal page | App screenshot/video showing inputs/stages/results | 7, 9 | planned |
| Comparison/evaluation web surface | Module 4 AGENTS/brief | Comparison page | Metrics/status/download evidence | 7, 8, 9 | planned |
| Theory web surface | Instructions Section 3; Questions web app | Theory page | Parts A-F visible in app and video | 6, 7, 9 | planned |
| README and execution instructions | Instructions Sections 1-2 | README.md | Clean install/run/test/experiment walkthrough | 1, 9, 10 | planned |
| Dependency declaration | Root AGENTS Python standards | pyproject.toml | Fresh environment install | 1, 10 | planned |
| Separate GitHub repository | Instructions Section 1 | Module_4 repository strategy | User-run Git status/remote and instructor access | 0, 10 | pending user action |
| Final PDF content | Instructions Section 5; brief | report docs/build script | Generated report audit and final PDF | 9, 10 | planned/pending user evidence |
| Digitized hand-worked/show-by-example content | Instructions Section 6 | typed theory/examples in report; user scans if needed | PDF inspection | 6, 9 | pending requirement review |
| Demonstration video | Instructions Section 4; brief | DEMO_VIDEO_CHECKLIST.md | User-recorded video showing app, inputs, processing, outputs | 9, 10 | pending user action |
| Google Classroom submission | Instructions Final Submission | user submission workflow | Uploaded PDF and video | 10 | pending user action |

## 28. Immediate next step: Phase 1 proposal

After the user reviews this plan and supplies the separate-repository Git status/branch information, I recommend proceeding with Phase 1 only:

1. create the Module 4 package/dependency/app skeleton;
2. define canonical types and safe image/mask IO;
3. create pending-safe data/results/documentation directories;
4. create the dashboard-compatible page-provider skeleton;
5. add foundational tests and a README skeleton; and
6. verify imports, tests, and app startup without adding segmentation, SAM2, or empirical results.

The required approval phrase is: Proceed with Phase 1.
