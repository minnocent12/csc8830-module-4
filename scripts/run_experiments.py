#!/usr/bin/env python3
"""Run explicitly configured Module 4 cases and write reproducible JSON/CSV records."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from module4.experiments import run_experiment_cases, write_csv_records, write_json_records  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="JSON file containing an explicit cases array")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Root for configured project-relative input paths")
    parser.add_argument("--output-json", type=Path, help="Optional JSON result path")
    parser.add_argument("--output-csv", type=Path, help="Optional CSV result path")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    configuration: Any = json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(configuration, dict) or not isinstance(configuration.get("cases"), list):
        raise ValueError("configuration must be a JSON object containing a cases array")
    records = run_experiment_cases(configuration["cases"], project_root=args.project_root)
    if args.output_json:
        write_json_records(records, args.output_json)
    if args.output_csv:
        write_csv_records(records, args.output_csv)
    if not args.output_json and not args.output_csv:
        print(json.dumps([record.to_dict() for record in records], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
