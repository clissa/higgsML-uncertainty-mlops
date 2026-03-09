"""Confidence interval construction from nonconformity scores.

Supports both *asymmetric* (two-sided quantile) and *central*
(symmetric ±q) interval types, parameterised by a significance level
``alpha``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Quantile extraction
# ---------------------------------------------------------------------------


def extract_quantiles(
    scores: np.ndarray,
    alpha: float,
    how: str = "diff",
    ci_type: str = "asymmetric",
) -> Tuple[float, float]:
    """Extract the quantile(s) needed to build a confidence interval.

    Parameters
    ----------
    scores : 1-D array
        Calibration nonconformity scores.
    alpha : float
        Significance level (e.g. ≈ 0.3173 for 1-sigma).
    how : str
        ``"diff"`` or ``"abs"`` — determines interpretation.
    ci_type : str
        ``"asymmetric"`` — different lower/upper quantiles.
        ``"central"`` — symmetric ±q(1 - alpha).

    Returns
    -------
    (q_low, q_high) : tuple of float
        Quantiles used in ``y_pred + q_low`` and ``y_pred + q_high``.
    """
    scores = np.asarray(scores, dtype=np.float64)

    if ci_type == "asymmetric":
        if how == "diff":
            q_low = float(np.percentile(scores, (alpha / 2) * 100))
            q_high = float(np.percentile(scores, (1 - alpha / 2) * 100))
        elif how == "abs":
            # For absolute scores the sign is always positive.
            # Asymmetric still uses two-sided quantiles of the absolute values.
            q_abs = float(np.percentile(scores, (1 - alpha / 2) * 100))
            q_low = -q_abs
            q_high = q_abs
        else:
            raise ValueError(f"Unknown how: {how!r}")

    elif ci_type == "central":
        if how == "diff":
            # Central: symmetric around median; use (1-alpha) quantile of
            # absolute deviations from median to keep it comparable.
            half_width = float(
                np.percentile(np.abs(scores - np.median(scores)), (1 - alpha) * 100)
            )
            q_low = float(np.median(scores)) - half_width
            q_high = float(np.median(scores)) + half_width
        elif how == "abs":
            q = float(np.percentile(scores, (1 - alpha) * 100))
            q_low = -q
            q_high = q
        else:
            raise ValueError(f"Unknown how: {how!r}")

    else:
        raise ValueError(f"Unknown ci_type: {ci_type!r}")

    return q_low, q_high


# ---------------------------------------------------------------------------
# Confidence interval construction
# ---------------------------------------------------------------------------


def compute_confidence_interval(
    y_pred: Union[float, np.ndarray],
    scores: np.ndarray,
    alpha: float,
    how: str = "diff",
    ci_type: str = "asymmetric",
) -> Tuple[np.ndarray, np.ndarray]:
    """Build confidence intervals from predictions and calibration scores.

    Parameters
    ----------
    y_pred : float or array
        Point prediction(s).
    scores : 1-D array
        Calibration nonconformity scores (for the model of interest).
    alpha : float
        Significance level.
    how, ci_type : str
        Passed to :func:`extract_quantiles`.

    Returns
    -------
    (lower, upper) : tuple of ndarray
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    q_low, q_high = extract_quantiles(scores, alpha, how=how, ci_type=ci_type)
    return y_pred + q_low, y_pred + q_high


# ---------------------------------------------------------------------------
# File-based convenience (loads scores from .npz)
# ---------------------------------------------------------------------------


def compute_confidence_intervals_from_file(
    y_pred: Union[float, np.ndarray],
    nonconf_scores_file: Path,
    model_name: str,
    alpha: float,
    how: str = "diff",
    ci_type: str = "asymmetric",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load saved nonconformity scores and build CIs.

    This is a convenience wrapper around
    :func:`compute_confidence_interval` for pipelines that persist scores
    to ``.npz`` files.
    """
    data = np.load(nonconf_scores_file)
    if model_name not in data:
        raise KeyError(
            f"Model {model_name!r} not found in {nonconf_scores_file}. "
            f"Available: {list(data.keys())}"
        )
    scores = data[model_name]
    return compute_confidence_interval(y_pred, scores, alpha, how=how, ci_type=ci_type)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def save_quantiles(
    quantiles: Dict[str, Tuple[float, float]],
    alpha: float,
    ci_type: str,
    how: str,
    output_dir: Path,
    filename: str = "calibration_quantiles.json",
) -> Path:
    """Save extracted quantiles and calibration parameters as JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    payload = {
        "alpha": alpha,
        "ci_type": ci_type,
        "how": how,
        "quantiles": {
            name: {"q_low": float(ql), "q_high": float(qh)}
            for name, (ql, qh) in quantiles.items()
        },
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    return path


def save_intervals(
    intervals_df: pd.DataFrame,
    output_dir: Path,
    filename: str = "confidence_intervals.csv",
) -> Path:
    """Persist CI table (model, experiment, y_pred, lower, upper, y_true)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    intervals_df.to_csv(path, index=False)
    return path
