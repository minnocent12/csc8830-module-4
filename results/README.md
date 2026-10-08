# Module 4 results

Phase 8 contains derived evidence from the six fixed AAU VAP cases listed in
[`data/experiment_manifest.json`](../data/experiment_manifest.json). The tracked files are small
and reproducible; raw inputs, binary source-mask conversions, and model checkpoints remain local
and ignored.

- `metrics/phase8_experiment_records.json` and `.csv`: serialized runner records.
- `metrics/phase8_experiment_summary.md`: generated per-case metrics and descriptive N=3 means.
- `metrics/phase8_artifacts.json`: artifact index with status and paths.
- `rgb/`: classical masks and boundary overlays.
- `thermal/`: classical masks, normalized intensity, bright/dark candidates, and overlays.
- `comparisons/`: canonical dataset references and green/red/blue TP/FP/FN overlays.
- `metrics/phase8_sam2_experiment_records.json`, `.csv`, `phase8_sam2_experiment_summary.md`:
  official SAM2 runs on the same six cases (classical vs SAM2, SAM2 vs ground truth).
- `rgb/*_sam2_mask.png`, `thermal/*_sam2_mask.png`: the recorded official SAM2 masks.
- `comparisons/*_classical_vs_sam2.png`, `*_sam2_vs_ground_truth.png`: SAM2 overlaps.
- `comparisons/*_side_by_side.png`: original, classical boundary, SAM2 boundary, and overlap per
  case (built by `scripts/build_sam2_comparison_figures.py` from the committed masks).
- `metrics/phase8_timing_records.json`, `phase8_timing_summary.md`: measured processing time of
  both methods and a check that rerunning both reproduces every recorded mask
  (`scripts/run_sam2_timing.py`).

Run the exporter from the repository root with:

```text
python scripts/run_phase8_evidence.py --manifest data/experiment_manifest.json \
  --project-root . --output-dir results
```

The exporter checks that serialized metrics match the exported masks. The SAM2 evidence and
timing commands are in the repository README and `docs/SAM2_COMPARISON.md`.
