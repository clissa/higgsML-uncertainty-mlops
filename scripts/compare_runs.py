#!/usr/bin/env python
"""Side-by-side comparison of metrics across recorded runs.

Reads the local run index (``results/runs_index.json`` by default) and
prints a table with one row per run and key metrics as columns.

Usage::

    # Print all runs
    python scripts/compare_runs.py

    # Filter by dataset and date, save to Markdown
    python scripts/compare_runs.py --dataset toy --since 2026-01-01 \\
        --output results/comparison.md

    # Custom metric columns
    python scripts/compare_runs.py --metrics roc_auc,coverage,ci_score

CLI arguments
-------------
--index PATH    Path to runs_index.json (default: results/runs_index.json).
--dataset STR   Filter to runs whose dataset matches STR.
--since DATE    ISO date (YYYY-MM-DD) — include only runs on or after this date.
--metrics LIST  Comma-separated metric suffixes to display.
                Default: roc_auc,coverage,ci_score,width.
--output PATH   Save comparison table to .csv or .md (inferred from extension).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Default settings
# ---------------------------------------------------------------------------

DEFAULT_INDEX = "results/runs_index.json"
DEFAULT_METRICS = ["accuracy", "coverage", "ci_score", "width"]

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def load_index(path: str) -> list:
    """Load the run index JSON, returning an empty list if the file is absent."""
    p = Path(path)
    if not p.exists():
        return []
    with open(p) as fh:
        data = json.load(fh)
    return data if isinstance(data, list) else []


def filter_runs(
    runs: list,
    dataset: Optional[str] = None,
    since: Optional[str] = None,
) -> list:
    """Apply optional filters to the run list."""
    result = runs

    if dataset:
        result = [r for r in result if r.get("dataset") == dataset]

    if since:
        try:
            cutoff = datetime.fromisoformat(since).replace(tzinfo=timezone.utc)
        except ValueError:
            print(f"Warning: could not parse --since date '{since}'. Ignoring filter.")
            cutoff = None
        if cutoff:
            filtered = []
            for r in result:
                ts = r.get("timestamp", "")
                try:
                    run_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if run_dt >= cutoff:
                        filtered.append(r)
                except (ValueError, AttributeError):
                    filtered.append(r)  # keep if unparseable
            result = filtered

    return result


def _aggregate_metric(flat_metrics: dict, suffix: str) -> Optional[float]:
    """Average all metric values whose key ends with *suffix*.

    Handles the ``{model_name}.{metric_suffix}`` naming convention used
    in the run index.  For example, suffix ``"roc_auc"`` matches
    ``"GLM.roc_auc"`` and ``"Random Forest.roc_auc"``.

    Special cases for calibration metrics which are stored as
    ``{model}.calib_coverage`` → suffix ``"coverage"``.
    """
    calib_map = {
        "coverage": "calib_coverage",
        "width": "calib_width",
        "mean_width": "calib_width",
        "ci_score": "calib_ci_score",
    }
    suffixes_to_try = [suffix]
    if suffix in calib_map:
        suffixes_to_try.append(calib_map[suffix])

    matched: List[float] = []
    for key, val in flat_metrics.items():
        for s in suffixes_to_try:
            if key == s or key.endswith(f".{s}"):
                if isinstance(val, (int, float)):
                    matched.append(float(val))
                break

    if not matched:
        return None
    return sum(matched) / len(matched)


def build_table(
    runs: list,
    metric_suffixes: Sequence[str],
) -> Tuple[List[str], List[List[str]]]:
    """Build headers + rows for the comparison table.

    Returns
    -------
    headers : list of str
    rows : list of list of str
    """
    headers = ["run_id", "timestamp", "dataset"] + list(metric_suffixes)
    rows = []
    for run in runs:
        ts = run.get("timestamp", "—")
        # Trim to date+time for readability
        try:
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime(
                "%Y-%m-%d %H:%M"
            )
        except (ValueError, AttributeError):
            pass

        flat_metrics = run.get("metrics", {})
        row: List[str] = [
            run.get("run_id", "—"),
            ts,
            run.get("dataset", "—"),
        ]
        for suffix in metric_suffixes:
            val = _aggregate_metric(flat_metrics, suffix)
            row.append(f"{val:.4f}" if val is not None else "—")
        rows.append(row)
    return headers, rows


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _tabulate_plain(headers: List[str], rows: List[List[str]]) -> str:
    """Simple f-string table — no external deps required."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(cells: List[str]) -> str:
        return "  ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells))

    sep = "  ".join("-" * w for w in col_widths)
    lines = [fmt_row(headers), sep] + [fmt_row(r) for r in rows]
    return "\n".join(lines)


def _tabulate_md(headers: List[str], rows: List[List[str]]) -> str:
    """GitHub-flavoured Markdown table."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(cells: List[str]) -> str:
        return "| " + " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells)) + " |"

    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    return "\n".join([fmt_row(headers), sep] + [fmt_row(r) for r in rows])


def _tabulate_csv(headers: List[str], rows: List[List[str]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(headers)
    writer.writerows(rows)
    return buf.getvalue()


def _try_tabulate(headers: List[str], rows: List[List[str]]) -> str:
    """Use ``tabulate`` if installed, otherwise fall back to plain text."""
    try:
        from tabulate import tabulate  # type: ignore

        return tabulate(rows, headers=headers, tablefmt="simple")
    except ImportError:
        return _tabulate_plain(headers, rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare metrics across runs recorded in runs_index.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--index",
        default=DEFAULT_INDEX,
        help=f"Path to runs_index.json (default: {DEFAULT_INDEX}).",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Filter runs by dataset name (e.g. 'toy' or 'higgs').",
    )
    parser.add_argument(
        "--since",
        default=None,
        metavar="YYYY-MM-DD",
        help="Include only runs on or after this date.",
    )
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help=(
            "Comma-separated metric suffixes to show.  "
            f"Default: {','.join(DEFAULT_METRICS)}."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save table to this file.  Extension determines format: .csv or .md.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    runs = load_index(args.index)
    if not runs:
        print(f"No runs found in '{args.index}'.")
        return

    runs = filter_runs(runs, dataset=args.dataset, since=args.since)
    if not runs:
        print("No runs match the given filters.")
        return

    metric_suffixes = [m.strip() for m in args.metrics.split(",") if m.strip()]
    headers, rows = build_table(runs, metric_suffixes)

    # Print to stdout
    print(f"\nComparing {len(runs)} run(s) — metrics: {', '.join(metric_suffixes)}\n")
    print(_try_tabulate(headers, rows))
    print()

    # Optionally save
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ext = out_path.suffix.lower()
        if ext == ".csv":
            content = _tabulate_csv(headers, rows)
        else:
            content = _tabulate_md(headers, rows)
        out_path.write_text(content, encoding="utf-8")
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
