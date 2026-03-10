"""Naming helper for structured wandb logging keys.

Enforces the ``Section/subsection/name`` taxonomy with five allowed
top-level sections.  This module is the single source of truth for all
wandb key construction.

Usage::

    from conformal_predictions.mlops.log_keys import wandb_key, EDA

    key = wandb_key(EDA, "plots", "target_distribution")
    # => "EDA/plots/target_distribution"
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
# Key builder
# ---------------------------------------------------------------------------


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
