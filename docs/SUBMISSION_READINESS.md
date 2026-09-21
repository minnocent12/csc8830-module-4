# Module 4 submission readiness audit

Audit date: 2026-09-21
Repository: [minnocent12/csc8830-module-4](https://github.com/minnocent12/csc8830-module-4)
Base audited: `main` at `7c129ef29d90351bcd9efe3498f6b73586adc0ef`
Audit branch: `task/module-4-final-submission-qa`

This document is a final submission-readiness audit, not a feature plan. It records what is
implemented, tested, experimentally validated, theoretically derived, blocked, and still
requires the student's manual action. No algorithm, experiment selection, measured value, or
SAM2 claim was changed during this audit.

## Status summary

| Area | Status | Evidence / note |
|---|---|---|
| Assignment traceability | PASS | Requirements below map the professor instructions/questions to source, app, report, and evidence. |
| Q1 RGB implementation and evidence | READY | Classical OpenCV pipeline, intermediate stages, mask, contour, overlay, fixed RGB records, report, and Streamlit page are present. |
| Q2 thermal implementation and evidence | READY | Classical intensity/Otsu/morphology/component pipeline, dual-polarity outputs, masks, contours, overlays, fixed thermal records, report, and Streamlit page are present. |
| Evaluation implementation and results | READY | IoU, Dice, precision, recall and edge conventions are tested and the six reference-backed records match the report. |
| Q3 Fourier Parts A–F | READY | Canonical theory, tested helpers, and Streamlit educational demonstrations are present. |
| Required PDF | READY | `deliverables/Module_4_Final_Report.pdf` exists, is 7 pages, rendered cleanly, and is the required Classroom artifact. |
| Editable report master | READY | `deliverables/Module_4_Final_Report.docx` exists, is 7 pages, and matches the PDF export. |
| Public repository | READY | GitHub reports the repository as public; `main` is synchronized and clean. |
| Experiment evidence | READY | Six fixed AAU VAP records, manifest, artifacts, and machine-readable metrics are traceable. |
| SAM2 actual comparison output | BLOCKED / USER ACTION REQUIRED | The assignment requires comparison with SAM2; the adapter is implemented, but official runtime/checkpoint inference has not been performed. |
| Demonstration script/checklist | READY | `docs/DEMO_SCRIPT.md` and `docs/DEMO_CHECKLIST.md` match the current app and disclose blocked SAM2 status. |
| Demonstration video | USER ACTION REQUIRED | The assignment requires a video; it has not been recorded. |
| Google Classroom submission | USER ACTION REQUIRED | Upload the final PDF and demonstration video manually. |

## Authoritative assignment requirements

The audit uses `MODULE_4_ASSIGNMENT_INSTRUCTIONS.md` and
`MODULE_4_ASSIGNMENT_QUESTIONS.md` as the sources of truth. No other project planning document
overrides them.

The instructions explicitly require a final PDF: Section 5 is titled “Final PDF Documentation,”
the checklist requires the final PDF to be uploaded to Google Classroom, and the final submission
list names “Final PDF documentation” and “Demonstration video.” The instructions do not specify a
video duration, file format, face/camera requirement, narration requirement, or hosting service.
The project’s 8–10 minute target in `docs/DEMO_SCRIPT.md` is a practical recording target, not a
professor-specified duration.

## Requirement traceability

| Requirement | Evidence / file | Status | Notes |
|---|---|---|---|
| Separate accessible GitHub repository | `https://github.com/minnocent12/csc8830-module-4`; GitHub repository visibility is public | PASS | Repository link is also included in the report. |
| Clearly documented source code and execution instructions | `README.md`, `pyproject.toml`, `src/module4/`, `scripts/` | PASS | Setup, app, tests, reproduction, and limitations are documented. |
| Working web application | `app.py`, `src/module4/webapp/`, standalone/root HTTP checks | PASS | Module 4 functionality is reachable through Streamlit pages. |
| Demonstration video showing app, inputs, processing, and outputs | `docs/DEMO_SCRIPT.md`, `docs/DEMO_CHECKLIST.md` | USER ACTION REQUIRED | Script/checklist are ready; the actual video is not recorded. |
| Final PDF documentation | `deliverables/Module_4_Final_Report.pdf` | READY | Seven-page PDF rendered from the Word master and visually inspected. |
| Problem description and implementation explanation | Report Sections 1–5 | PASS | Covers objective, RGB, thermal, SAM2 role, and evaluation. |
| Methods and algorithms | `docs/RGB_PIPELINE.md`, `docs/THERMAL_PIPELINE.md`, report Sections 2–3 | PASS | Classical OpenCV stages and parameters are described. |
| Examples, results, figures, and visualizations | `results/`, report Sections 6–7, five figures and two tables | PASS | Figures are tracked generated artifacts; the table is data-driven. |
| GitHub link in final PDF | Report Repository section and hyperlink relationship | PASS | Link resolves to the public Module 4 repository. |
| Any required hand-worked/show-by-example material | Assignment docs; Fourier explanation in report and `docs/FOURIER_THEORY.md` | NOT REQUIRED | No separate handwritten worksheet is specified in the Module 4 requirements. |
| PDF and video Classroom submission | Assignment Instructions Section 5 and Final Submission | USER ACTION REQUIRED | Repository work is complete; Classroom upload is not performed by this audit. |

## Question 1 RGB traceability

| Q1 requirement | Source implementation | App/report/evidence | Status |
|---|---|---|---|
| Standard RGB/color input | `src/module4/io_utils.py`, `src/module4/rgb.py` | RGB page accepts BGR-decoded color input; report Section 2 | PASS |
| Classical computer vision only | `src/module4/rgb.py`, `src/module4/preprocessing.py`, `src/module4/components.py` | GrabCut, morphology, ROI component selection; no ML/DL primary segmentation | PASS |
| Human region segmentation | `run_rgb_segmentation` | `results/rgb/*_classical_mask.png`; report Section 2 | PASS |
| Intermediate processing | `RGBSegmentationResult` fields and RGB page stages | RGB page shows ROI, raw GrabCut, cleanup, component, and final stages | PASS |
| Final binary human mask | `run_rgb_segmentation(...).final_mask` | Tracked per-case masks and report result table | PASS |
| Boundary/contour | `src/module4/contours.py`, `extract_external_contours` | Boundary overlays in `results/rgb/`; report Figure 1 | PASS |
| Final overlay | `boundary_overlay_bgr` and Streamlit display | `results/rgb/*_boundary_overlay.png`; report Figure 1 | PASS |
| SAM2 comparison role | `src/module4/reference/sam2_reference.py`, `docs/SAM2_COMPARISON.md` | Report Section 4 states SAM2 is reference-only and blocked | BLOCKED / USER ACTION REQUIRED for actual SAM2 output |
| Visual/numerical evaluation where references exist | `src/module4/metrics.py`, Phase 8 records | Report Sections 5–7; RGB records have ground-truth metrics | READY |

The RGB smoke test used fixed manifest case 00085 and reproduced IoU `0.4895`, a non-empty
binary mask, contours, and a boundary overlay without changing parameters.

## Question 2 thermal traceability

| Q2 requirement | Source implementation | App/report/evidence | Status |
|---|---|---|---|
| Thermal/infrared input | `src/module4/io_utils.py`, `src/module4/thermal.py` | Fixed AAU VAP thermal JPG cases and Thermal page | PASS |
| Classical computer vision only | `src/module4/thermal.py`, `src/module4/preprocessing.py` | Intensity, Otsu, morphology, connected components, geometry scoring | PASS |
| Source/intensity handling | `_source_to_intensity`, `normalize_thermal_intensity` | Report Section 3 and thermal warnings | PASS |
| Thresholding/Otsu | `run_thermal_segmentation` bright and dark `cv2.threshold(... OTSU)` calls | Thermal page shows both candidates; report Section 3 | PASS |
| Morphology | `apply_morphological_cleanup` | Parameters recorded in six records and report | PASS |
| Connected components/region selection | `_component_candidates`, `_best_component` | Selected component and warnings are exposed in thermal result | PASS |
| Final binary human mask | `ThermalSegmentationResult.final_mask` | `results/thermal/*_classical_mask.png`; report table | PASS |
| Boundary/contour | `extract_external_contours` and `boundary_overlay_bgr` | Thermal overlays and report Figure 4 | PASS |
| Comparison/evaluation | `metrics.py`, comparison page, Phase 8 records | Report Sections 5–7 and thermal overlap figures | READY |
| Accurate thermal wording | `docs/THERMAL_PIPELINE.md`, report Section 3 | False-color JPG is explicitly not calibrated temperature; no claim that a person is always hotter | PASS |
| SAM2 comparison role | Isolated adapter and comparison page | Report Section 4 honestly records blocked real inference | BLOCKED / USER ACTION REQUIRED for actual SAM2 output |

The thermal smoke test used fixed manifest case 00085, reproduced IoU `0.0417`, selected dark
polarity, produced a binary mask and contours, and retained the false-color/intensity caveat.

## SAM2 requirement determination

The assignment explicitly requires the traditional result to be compared against SAM2 segmentation
for both Q1 and Q2, and lists a SAM2 segmentation result and comparison as expected output. The
assignment does not state a separate required SAM2 model, checkpoint, runtime, timing, or metric
format. Therefore:

- SAM2 integration and reference semantics are implemented and tested.
- Actual SAM2 segmentation output is not available in the current environment because the
  official package, PyTorch runtime, and checkpoint are absent.
- The current report correctly includes no SAM2 mask, metric, timing, or screenshot.
- If the instructor expects the explicit SAM2 result listed in the expected output, the student
  must run the official SAM2 environment and add real provenance-bearing comparison evidence before
  final submission. This is **USER ACTION REQUIRED / BLOCKED**, not a completed empirical result.
- SAM2 is never treated as ground truth; the Phase 8 metrics use available AAU VAP ground-truth
  masks instead.

## Evaluation traceability

| Evaluation requirement | Evidence | Status |
|---|---|---|
| IoU | `src/module4/metrics.py`, tests, report formula/table, six JSON records | PASS |
| Dice | `src/module4/metrics.py`, tests, report formula/table, six JSON records | PASS |
| Precision | `src/module4/metrics.py`, tests, report formula/table, six JSON records | PASS |
| Recall | `src/module4/metrics.py`, tests, report formula/table, six JSON records | PASS |
| Empty-mask and zero-denominator conventions | `docs/EXPERIMENTAL_RESULTS.md`, `metrics.py`, tests, report Section 5 | PASS |
| Valid reference semantics | `src/module4/validation.py`, `reference_type`, `reference_status`, alignment metadata | PASS |
| No fabricated unavailable-reference scores | Pending/failed reference logic and no SAM2 metrics in records | PASS |
| Ground truth distinct from SAM2 reference | `docs/SAM2_COMPARISON.md`, report Section 4, record fields | PASS |
| Report values match machine-readable records | `results/metrics/phase8_experiment_records.json` cross-check | PASS |

The six records are three RGB and three thermal cases from fixed frames 00085, 00135, and 00185.
RGB IoU values are `0.4895`, `0.3607`, and `0.4183`; thermal values are `0.0417`, `0.0206`, and
`0.0210`. All displayed Dice, precision, and recall values were programmatically matched against
the JSON records.

## Question 3 Fourier Parts A–F

| Part | Required content | Evidence | Status |
|---|---|---|---|
| A | Continuous 2D transform, DFT, inverse, magnitude, phase, low/high frequencies, `fftshift` | `docs/FOURIER_THEORY.md` Part A; `src/module4/fourier.py`; report Part A; Fourier page | PASS |
| B | Edges as rapid changes and high-frequency explanation | Theory Part B; report Part B; Fourier page | PASS |
| C | High-pass filtering and `H_HP = 1 − H_LP` | Theory Part C; `gaussian_high_pass`; report Part C; Fourier page | PASS |
| D | Derivative property for x and y | Theory Part D; `frequency_derivative(axis="x"/"y")`; report Part D; Fourier page | PASS |
| E | Laplacian derivation and transfer function | Theory Part E; `frequency_laplacian`; report Part E; Fourier page | PASS |
| F | Frequency-selective segmentation, global localization limitation, local/windowed analysis, advantages, limitations | Theory Part F; `local_frequency_energy`; report Part F; Fourier page | PASS |

## Report and PDF QA

| Check | Result |
|---|---|
| Word master exists and opens | PASS — `deliverables/Module_4_Final_Report.docx` |
| PDF exists and opens | PASS — `deliverables/Module_4_Final_Report.pdf` |
| DOCX/PDF page count | PASS — 7 pages each |
| Images render | PASS — five figures inspected |
| Tables fit margins | PASS — two tables inspected |
| Equations render/read | PASS — Fourier expressions inspected |
| Blank/corrupted pages | PASS — none found |
| Captions/headings | PASS — consistent and readable |
| Private absolute paths | PASS — none found in document or tracked content |
| Revision marks/comments/temporary content | PASS — none found |
| Hyperlinks | PASS — repository, AAU, Kaggle, and SAM2 relationships present |
| Accessibility | PASS — document accessibility audit reported zero findings |

The PDF was generated from the reviewed DOCX and matches its content, metric values, figures,
tables, and SAM2 status. The DOCX remains the editable/master copy.

## Repository and README audit

| Check | Status | Evidence |
|---|---|---|
| Public repository | PASS | GitHub visibility query reports `isPrivate: false`. |
| README purpose and questions | PASS | README identifies RGB, thermal, evaluation, SAM2, and Fourier work. |
| Setup/dependencies | PASS | README and `pyproject.toml`. |
| App/test commands | PASS | README documents `streamlit run app.py` and `python -m pytest -q`. |
| Experiment reproduction | PASS | README and `docs/EXPERIMENTAL_RESULTS.md` point to manifest/exporter. |
| Final report location | PASS | README now links the DOCX, PDF, and this audit. |
| SAM2 limitation | PASS | README and docs disclose blocked real inference. |
| Tracked secrets/checkpoints/raw dumps | PASS | Hygiene and sensitive-file scans produced no findings. |
| Internal instruction files | PASS | Course-local instruction files remain ignored/untracked. |

## Experiment reproducibility

The trace is:

`data/experiment_manifest.json` → `scripts/run_phase8_evidence.py` →
`module4.experiments` and existing RGB/thermal pipelines → `results/metrics/` and `results/`.

The manifest fixes the AAU VAP source, frame selection rule, ROIs, parameters, and per-case RNG
seeds. The six JSON records have matching image IDs, modalities, available ground-truth references,
and 480 × 640 dimensions. Referenced masks, overlays, and metrics files exist. A fixed-case smoke
run reproduced the 00085 RGB and thermal IoU values without changing selection or writing new
tracked results.

## Application QA

The standalone Module 4 app and shared/root dashboard each returned HTTP 200 for `/` and
`/_stcore/health`. The page-provider contract exposes these four pages:

1. RGB Human Boundary
2. Thermal Human Boundary
3. Comparison and Evaluation
4. Fourier Theory

The representative fixed-case smoke workflow passed RGB segmentation/reference evaluation, thermal
dual-polarity/reference evaluation, comparison metric formatting, and Fourier helper execution.
The browser-control runtime was unavailable for this audit, so no claim is made that a browser click
sequence was recorded. The app should be clicked through once during the user's final video setup.

## Demonstration video requirement and status

The assignment requires a video showing the application running, relevant inputs, implemented
functionality, processing, and resulting outputs. It permits a screen recording or properly
captured footage. It does not specify duration, file format, face/camera, or narration requirements.
The video has not been recorded, so this requirement is **USER ACTION REQUIRED**.

### Final recording checklist

1. Start at the Module 4 app home/page selector and state the objective.
2. On `RGB Human Boundary`, use fixed Scene 1 frame `00085`, show the ROI, intermediate stages,
   final mask, contour, and boundary overlay.
3. On `Thermal Human Boundary`, use fixed Scene 1 frame `00085`, show normalized intensity, both
   Otsu polarity candidates, selected dark polarity, warnings, final mask, and overlay.
4. On `Comparison and Evaluation`, show a valid binary reference and the IoU/Dice/precision/recall
   display. Explain that Phase 8 metrics use AAU VAP ground-truth masks.
5. Show the optional SAM2 control/status honestly. State that real inference is blocked and do not
   display fabricated SAM2 output or metrics.
6. Open `Fourier Theory` and briefly show Parts A–F, including the educational local-frequency
   demonstration.
7. State the fixed experiment scope, repository URL, false-color thermal caveat, and limitations.
8. Stop recording, play the complete file back, and verify that inputs, processing, outputs, audio
   if used, and SAM2 wording are clear.
9. Upload the final video through the required Google Classroom assignment.

The project script targets approximately 8–10 minutes for concision; this is a project guideline,
not an assignment-mandated duration.

## Submission inventory

### Required submission

- `deliverables/Module_4_Final_Report.pdf` — READY; upload to Google Classroom.
- Demonstration video — USER ACTION REQUIRED; record and upload to Google Classroom.
- Public repository URL — READY; include/use `https://github.com/minnocent12/csc8830-module-4`.

### Supporting artifacts

- `deliverables/Module_4_Final_Report.docx` — editable/master report.
- `README.md` — setup, app, tests, experiments, limitations, and artifact links.
- `app.py` and `src/module4/webapp/` — standalone Streamlit app and page provider.
- `results/metrics/phase8_experiment_records.json` and `.csv` — machine-readable evidence.
- `results/metrics/phase8_experiment_summary.md` and `phase8_artifacts.json` — generated summary/provenance.
- `results/rgb/`, `results/thermal/`, `results/comparisons/` — derived evidence images.
- `docs/RGB_PIPELINE.md`, `docs/THERMAL_PIPELINE.md`, `docs/SAM2_COMPARISON.md`,
  `docs/FOURIER_THEORY.md`, `docs/EXPERIMENTAL_RESULTS.md` — technical documentation.
- `docs/DEMO_SCRIPT.md` and `docs/DEMO_CHECKLIST.md` — recording preparation.

### Google Classroom checklist

1. Confirm the filename `Module_4_Final_Report.pdf` and open it once more.
2. Upload the PDF to the correct Module 4 assignment.
3. Confirm the repository URL is visible in the PDF and add it where Classroom requests it.
4. Record the demonstration video using the checklist above.
5. Upload the video or its required link to the same Classroom assignment.
6. Confirm both attachments/links are present and readable.
7. Submit the assignment and verify the submission status.
8. Save the confirmation if desired.

This audit does not claim that any Google Classroom upload or submission has occurred.

## Final classification

| Category | Status |
|---|---|
| Code | READY |
| Tests | READY |
| Web app | READY |
| Experiment evidence | READY for the fixed ground-truth subset |
| Word report | READY |
| Required PDF | READY |
| GitHub repository | READY |
| Demo script | READY |
| Demo video | USER ACTION REQUIRED |
| Google Classroom submission | USER ACTION REQUIRED |
| SAM2 real inference | BLOCKED / USER ACTION REQUIRED because the assignment expects a SAM2 comparison output and the runtime is unavailable |
| Overall submission | **NOT READY — REQUIRED ISSUES REMAIN** |

The remaining issues are manual/external: record and validate the required video, complete the
Google Classroom upload, and, if the instructor requires the expected SAM2 segmentation result,
run the official SAM2 environment and add genuine provenance-bearing comparison evidence. No Phase
10 algorithm or experiment changes were made.
