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

Run the exporter from the repository root with:

```text
python scripts/run_phase8_evidence.py --manifest data/experiment_manifest.json \
  --project-root . --output-dir results
```

The exporter checks that serialized metrics match the exported masks. No SAM2 mask, timing
number, or SAM2 metric is included because the official runtime was blocked in this environment.
