#!/usr/bin/env python
"""List past runs from the local run index.

Usage::

    python scripts/list_runs.py
    python scripts/list_runs.py --index results/runs_index.json
    python scripts/list_runs.py --dataset toy
    python scripts/list_runs.py --since 2026-01-01

Output columns: run_id, timestamp, dataset, output_dir, key metrics.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure the src package is importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from conformal_predictions.mlops.run_index import load_index

# Key metrics to display — full slash-key taxonomy used by the tracker.
# Format: Section/subsection/name (e.g. "Evaluation/test/accuracy").
_DISPLAY_METRICS = [
    "Evaluation/test/accuracy",
    "Evaluation/test/f1",
    "Calibration/metrics/coverage",
    "Calibration/metrics/ci_score",
]

DEFAULT_INDEX = "results/runs_index.json"


def _pick_metric(metrics: dict, key: str) -> str:
    """Return the metric value for the given slash-key, formatted as a string.

    Accepts the full slash-key (e.g. "Evaluation/test/accuracy") or a bare
    name (e.g. "accuracy") for exact match only.  The old dotted-model-prefix
    convention is no longer supported.
    """
    if key in metrics:
        v = metrics[key]
        return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
    return "—"


def _format_table(rows: list[list[str]], headers: list[str]) -> str:
    """Format rows + headers as a plain-text table."""
    try:
        from tabulate import tabulate  # type: ignore[import]

        return tabulate(rows, headers=headers, tablefmt="simple")
    except ImportError:
        pass

    # Fallback: plain f-string formatting
    all_rows = [headers] + rows
    widths = [max(len(str(r[i])) for r in all_rows) for i in range(len(headers))]
    sep = "  ".join("-" * w for w in widths)
    lines = []
    lines.append("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    lines.append(sep)
    for row in rows:
        lines.append("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List past runs from the local run index.",
    )
    parser.add_argument(
        "--index",
        type=str,
        default=DEFAULT_INDEX,
        help=f"Path to runs_index.json (default: {DEFAULT_INDEX}).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Filter runs by dataset name.",
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Only show runs on or after this date (UTC).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    records = load_index(args.index)
    if not records:
        print(f"No runs found in {args.index}.")
        return

    # --- filter ---
    if args.dataset:
        records = [r for r in records if r.get("dataset") == args.dataset]

    if args.since:
        try:
            since_dt = datetime.strptime(args.since, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            print(f"Invalid --since date: {args.since!r} (expected YYYY-MM-DD)")
            sys.exit(1)
        records = [
            r
            for r in records
            if datetime.fromisoformat(r.get("timestamp", "1970-01-01T00:00:00+00:00"))
            >= since_dt
        ]

    if not records:
        print("No runs match the filters.")
        return

    # --- build table ---
    metric_keys = [
        "Evaluation/test/accuracy",
        "Evaluation/test/f1",
        "Calibration/metrics/coverage",
        "Calibration/metrics/ci_score",
    ]
    headers = ["run_id", "timestamp", "dataset", "model", "output_dir"] + metric_keys

    rows = []
    for r in records:
        metrics = r.get("metrics", {})
        row = [
            r.get("run_id", "?")[:8],
            r.get("timestamp", "?")[:19],
            r.get("dataset", "?"),
            r.get("model_name", "?"),
            str(r.get("output_dir", "?"))[-40:],
        ] + [_pick_metric(metrics, k) for k in metric_keys]
        rows.append(row)

    print(f"\nRuns index: {args.index}  ({len(records)} run(s))\n")
    print(_format_table(rows, headers))
    print()


if __name__ == "__main__":
    main()
