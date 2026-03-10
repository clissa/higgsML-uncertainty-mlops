"""Per-run Markdown report generator.

Produces a human-readable ``report.md`` that summarises run metadata,
evaluation metrics, calibration quality, and references all saved plot
artifacts.  The generator degrades gracefully: missing sections are
omitted rather than raising exceptions.

Usage::

    from conformal_predictions.evaluation.reports import generate_run_report
    report_path = generate_run_report(
        ctx=run_ctx,
        metrics=eval_results,
        calibration_results=calib_results,
        output_path=ctx.output_dir / "report.md",
    )
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Union

if TYPE_CHECKING:
    from conformal_predictions.mlops.run_context import RunContext

PathLike = Union[str, Path]

# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------


def _md_table(headers: list[str], rows: list[list]) -> str:
    """Render a plain Markdown table."""
    col_widths = [len(h) for h in headers]
    str_rows = []
    for row in rows:
        str_row = [str(c) for c in row]
        str_rows.append(str_row)
        for i, cell in enumerate(str_row):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(cells: list[str]) -> str:
        return (
            "| "
            + " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells))
            + " |"
        )

    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    lines = [fmt_row(headers), sep] + [fmt_row(r) for r in str_rows]
    return "\n".join(lines)


def _fmt(value: object, decimals: int = 4) -> str:
    """Format a cell value as a rounded float string or plain string."""
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value) if value is not None else "—"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_run_report(
    ctx: "RunContext",
    metrics: Optional[Dict[str, dict]] = None,
    calibration_results: Optional[Dict[str, dict]] = None,
    output_path: Optional[PathLike] = None,
) -> Path:
    """Write a Markdown summary report for one pipeline run.

    Parameters
    ----------
    ctx : RunContext
        The active run context (provides metadata and artifact list).
    metrics : dict, optional
        ``{model_name: {"performance": {...}, "calibration": {...}}}`` as
        returned by ``Trainer.evaluate()``.  When *None* or empty, the
        metrics sections are omitted.
    calibration_results : dict, optional
        Alias / override for the ``"calibration"`` sub-dict inside
        *metrics*.  When provided it takes precedence.
    output_path : str or Path, optional
        Destination for the ``.md`` file.  Defaults to
        ``ctx.output_dir / "report.md"``.

    Returns
    -------
    Path
        The absolute path of the written report file.
    """
    if output_path is None:
        output_path = ctx.output_dir / "report.md"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    # ---- title ----
    lines.append(f"# Run Report — `{ctx.run_id}`\n")
    lines.append(
        f"*Generated on {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC*\n"
    )

    # ---- metadata block ----
    lines.append("## Run Metadata\n")
    config_path = getattr(ctx, "config_path", None) or "—"
    git_commit = getattr(ctx, "git_commit", None) or "—"
    lines += [
        "| Field         | Value |",
        "|---------------|-------|",
        f"| **Run ID**    | `{ctx.run_id}` |",
        f"| **Timestamp** | {ctx.timestamp} |",
        f"| **Dataset**   | {ctx.dataset} |",
        f"| **Git Commit**| `{git_commit}` |",
        f"| **Config**    | `{config_path}` |",
        f"| **Output Dir**| `{ctx.output_dir}` |",
        "",
    ]

    # ---- performance metrics table ----
    perf_rows = _build_perf_rows(metrics)
    if perf_rows:
        lines.append("## Classification Metrics (test set, avg over experiments)\n")
        headers = [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC",
            "PR-AUC",
        ]
        rows = [
            [
                model,
                _fmt(row.get("accuracy")),
                _fmt(row.get("precision")),
                _fmt(row.get("recall")),
                _fmt(row.get("f1")),
                _fmt(row.get("roc_auc")),
                _fmt(row.get("pr_auc")),
            ]
            for model, row in perf_rows.items()
        ]
        lines.append(_md_table(headers, rows))
        lines.append("")

    # ---- calibration quality table ----
    calib_rows = _build_calib_rows(metrics, calibration_results)
    if calib_rows:
        lines.append("## Calibration Quality Metrics\n")
        headers = ["Model", "Coverage", "Mean Width", "CI Score"]
        rows = [
            [
                model,
                _fmt(row.get("coverage")),
                _fmt(row.get("width")),
                _fmt(row.get("ci_score")),
            ]
            for model, row in calib_rows.items()
        ]
        lines.append(_md_table(headers, rows))
        lines.append("")

    # ---- plot references (organised by section) ----
    plot_artifacts = [a for a in ctx.artifacts if a.get("type") == "plot"]
    ea_artifacts = [a for a in ctx.artifacts if a.get("type") == "error_analysis"]

    # Group plots by category
    eda_plots = [
        a
        for a in plot_artifacts
        if "target_distribution" in a["path"] or "contour" in a["path"]
    ]
    eval_plots = [
        a
        for a in plot_artifacts
        if any(k in a["path"] for k in ("roc_curve", "pr_curve", "predictions_ecdf"))
    ]
    calib_plots = [
        a
        for a in plot_artifacts
        if any(
            k in a["path"]
            for k in ("nonconformity", "mu_hat", "ci_", "q_low", "q_high", "block_ci")
        )
    ]
    other_plots = [
        a for a in plot_artifacts if a not in eda_plots + eval_plots + calib_plots
    ]

    if eda_plots:
        lines.append("## EDA Plots\n")
        for art in eda_plots:
            rel_path = art["path"]
            desc = art.get("description") or rel_path
            lines.append(f"### {desc}\n")
            lines.append(f"![{desc}]({rel_path})\n")

    if eval_plots:
        lines.append("## Evaluation Plots\n")
        for art in eval_plots:
            rel_path = art["path"]
            desc = art.get("description") or rel_path
            lines.append(f"### {desc}\n")
            lines.append(f"![{desc}]({rel_path})\n")

    if calib_plots:
        lines.append("## Calibration Plots\n")
        for art in calib_plots:
            rel_path = art["path"]
            desc = art.get("description") or rel_path
            lines.append(f"### {desc}\n")
            lines.append(f"![{desc}]({rel_path})\n")

    if other_plots:
        lines.append("## Other Plots\n")
        for art in other_plots:
            rel_path = art["path"]
            desc = art.get("description") or rel_path
            lines.append(f"### {desc}\n")
            lines.append(f"![{desc}]({rel_path})\n")

    if ea_artifacts:
        lines.append("## Error Analysis Artifacts\n")
        for art in ea_artifacts:
            rel_path = art["path"]
            desc = art.get("description") or rel_path
            lines.append(f"- [{desc}]({rel_path})\n")

    # ---- footer ----
    lines.append("---\n")
    lines.append(f"*Run output directory: `{ctx.output_dir}`*\n")

    content = "\n".join(lines)
    output_path.write_text(content, encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_perf_rows(metrics: Optional[Dict[str, dict]]) -> Dict[str, dict]:
    """Extract per-model performance dicts from evaluate() results."""
    if not metrics:
        return {}
    rows: Dict[str, dict] = {}
    for model_name, entry in metrics.items():
        perf = entry.get("performance", {})
        if perf:
            rows[model_name] = perf
    return rows


def _build_calib_rows(
    metrics: Optional[Dict[str, dict]],
    calibration_results: Optional[Dict[str, dict]],
) -> Dict[str, dict]:
    """Extract per-model calibration quality dicts."""
    rows: Dict[str, dict] = {}
    # Prefer explicit calibration_results override
    if calibration_results:
        for model_name, cal in calibration_results.items():
            if cal:
                rows[model_name] = cal
        return rows
    # Fall back to nested "calibration" key inside metrics
    if metrics:
        for model_name, entry in metrics.items():
            cal = entry.get("calibration", {})
            if cal:
                rows[model_name] = cal
    return rows
