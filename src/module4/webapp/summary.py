"""Optional Home-page summary for the combined CSc 8830 course dashboard.

The combined dashboard may call ``get_module_summary()`` to show one status chip on this
module's Home card. The standalone app never uses it.

Evidence contract: the summary reports "Results available" only when the committed Phase 8
experiment records file is a non-empty list and every record states
``"classical_status": "completed"``. Otherwise it returns ``None`` and no chip is shown. It
only reads that small committed file: it never runs OpenCV or SAM2, recomputes metrics,
judges result quality, writes files, or touches the network.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

EXPERIMENT_RECORDS = "results/metrics/phase8_experiment_records.json"


@dataclass(frozen=True)
class ModuleSummary:
    """One status chip: ``kind`` is a shared chip kind (neutral, info, success, warning, error)."""

    label: str
    kind: str


RESULTS_AVAILABLE = ModuleSummary("Results available", "success")


def get_module_summary(repo_root: Path | None = None) -> ModuleSummary | None:
    """Return the Home status for this module, or ``None`` when the evidence is incomplete."""
    path = (repo_root or _REPO_ROOT) / EXPERIMENT_RECORDS
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(records, list) or not records:
        return None
    if not all(isinstance(r, dict) and r.get("classical_status") == "completed" for r in records):
        return None
    return RESULTS_AVAILABLE
