"""Naming helper for structured wandb logging keys.

Enforces the ``Section/subsection/name`` taxonomy with five allowed
top-level sections.  This module is the single source of truth for all
wandb key construction.

Usage::

    from conformal_predictions.mlops.log_keys import wandb_key, calib_key, plots_key, EDA

    key = wandb_key(EDA, "plots", "target_distribution")
    # => "EDA/plots/target_distribution"

    flat_calib = calib_key("coverage")
    # => "Calibration/coverage"

    flat_plot = plots_key("test_roc_curve")
    # => "Plots/test_roc_curve"
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Section constants
# ---------------------------------------------------------------------------

EDA: str = "EDA"
EVALUATION: str = "Evaluation"
CALIBRATION: str = "Calibration"
ERROR_ANALYSIS: str = "ErrorAnalysis"
PLOTS: str = "Plots"

_ALLOWED_SECTIONS = frozenset({EDA, EVALUATION, CALIBRATION, ERROR_ANALYSIS, PLOTS})


# ---------------------------------------------------------------------------
# Key builders
# ---------------------------------------------------------------------------


def calib_key(name: str) -> str:
    """Build a flat ``Calibration/<name>`` wandb metric key.

    Use this for all Calibration-section metrics to keep the wandb
    dashboard free of nested sub-sections.

    Parameters
    ----------
    name : str
        Metric or artifact name, e.g. ``"q_low"`` or ``"coverage"``.

    Returns
    -------
    str
        Key in the form ``"Calibration/<name>"``.
    """
    return f"Calibration/{name}"


def plots_key(name: str) -> str:
    """Build a flat ``Plots/<name>`` wandb image key.

    Use this for all Plots-section images to keep the wandb dashboard
    free of nested sub-sections.

    Parameters
    ----------
    name : str
        Plot artifact name, e.g. ``"test_roc_curve"`` or ``"predictions_ecdf"``.

    Returns
    -------
    str
        Key in the form ``"Plots/<name>"``.
    """
    return f"Plots/{name}"


def wandb_key(section: str, subsection: str, name: str) -> str:
    """Build a ``Section/subsection/name`` wandb metric key.

    Parameters
    ----------
    section : str
        One of ``EDA``, ``Evaluation``, ``Calibration``, ``ErrorAnalysis``,
        ``Plots``.
    subsection : str
        Logical grouping within the section (e.g. ``"plots"``, ``"train"``).
    name : str
        Metric or artifact name.

    Returns
    -------
    str
        Key in the form ``"Section/subsection/name"``.

    Raises
    ------
    AssertionError
        If *section* is not in the allowed set.
    """
    assert section in _ALLOWED_SECTIONS, (
        f"Invalid section {section!r}. " f"Allowed: {sorted(_ALLOWED_SECTIONS)}"
    )
    return f"{section}/{subsection}/{name}"
