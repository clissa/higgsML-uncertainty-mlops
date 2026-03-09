"""Performance and calibration quality metrics.

Provides a simple registry-based approach: ``METRIC_REGISTRY`` maps
metric names to callables with a uniform ``(y_true, y_pred, y_proba)``
signature.  Higher-level functions compose these into summary dicts.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

MetricFn = Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float]


def _loss(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray]
) -> float:
    """Binary cross-entropy loss (requires probabilities)."""
    if y_proba is None:
        return float("nan")
    return float(log_loss(y_true, y_proba))


def _accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray]
) -> float:
    return float(accuracy_score(y_true, y_pred))


def _precision(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray]
) -> float:
    return float(precision_score(y_true, y_pred, zero_division=0))


def _recall(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray]
) -> float:
    return float(recall_score(y_true, y_pred, zero_division=0))


def _f1(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray]) -> float:
    return float(f1_score(y_true, y_pred, zero_division=0))


def _pr_auc(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray]
) -> float:
    if y_proba is None:
        return float("nan")
    try:
        return float(average_precision_score(y_true, y_proba))
    except ValueError:
        return float("nan")


def _roc_auc(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray]
) -> float:
    if y_proba is None:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_proba))
    except ValueError:
        return float("nan")


METRIC_REGISTRY: Dict[str, MetricFn] = {
    "loss": _loss,
    "accuracy": _accuracy,
    "precision": _precision,
    "recall": _recall,
    "f1": _f1,
    "pr_auc": _pr_auc,
    "roc_auc": _roc_auc,
}


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------


def compute_performance_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    metric_names: Optional[Sequence[str]] = None,
    loss_fn_name: str = "binary_cross_entropy",
) -> Dict[str, Union[float, str]]:
    """Compute classification performance metrics.

    Parameters
    ----------
    y_true : 1-D array
        Ground truth labels.
    y_pred : 1-D array
        Hard predictions (after threshold).
    y_proba : 1-D array, optional
        Predicted class-1 probabilities (needed for loss, PR-AUC, ROC-AUC).
    metric_names : sequence of str, optional
        Which metrics to compute.  Defaults to all registered metrics.
    loss_fn_name : str
        Descriptive name for the loss function stored in the output dict.

    Returns
    -------
    dict
        ``{"loss_name": ..., "loss": ..., "accuracy": ..., ...}``
    """
    if metric_names is None:
        metric_names = list(METRIC_REGISTRY.keys())

    result: Dict[str, Union[float, str]] = {"loss_name": loss_fn_name}
    for name in metric_names:
        fn = METRIC_REGISTRY.get(name)
        if fn is None:
            raise KeyError(
                f"Unknown metric {name!r}. Available: {list(METRIC_REGISTRY)}"
            )
        result[name] = fn(np.asarray(y_true), np.asarray(y_pred), y_proba)

    return result


# ---------------------------------------------------------------------------
# Calibration quality metrics
# ---------------------------------------------------------------------------


def compute_coverage(
    lower: np.ndarray,
    upper: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """Empirical coverage: fraction of true values inside [lower, upper]."""
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    return float(np.mean((lower < y_true) & (y_true < upper)))


def compute_mean_width(
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Mean confidence interval width."""
    return float(np.mean(np.asarray(upper) - np.asarray(lower)))


def compute_ci_score(
    coverage: float,
    width: float,
    alpha: float,
    lambda_: float = 1.0,
    epsilon: float = 1e-6,
) -> float:
    """Challenge-style CI score.

    .. math::

        s = -\\log\\bigl((w + \\varepsilon) \\cdot (1 + \\lambda |c - c_0|)\\bigr)

    where :math:`c_0 = 1 - \\alpha`.
    """
    c0 = 1.0 - alpha
    penalty = 1.0 + lambda_ * abs(coverage - c0)
    return -math.log((width + epsilon) * penalty)


def compute_calibration_metrics(
    lower: np.ndarray,
    upper: np.ndarray,
    y_true: np.ndarray,
    alpha: float,
    lambda_: float = 1.0,
    epsilon: float = 1e-6,
) -> Dict[str, float]:
    """Compute all calibration quality metrics in one call.

    Returns
    -------
    dict
        ``{"coverage": ..., "width": ..., "ci_score": ...}``
    """
    cov = compute_coverage(lower, upper, y_true)
    w = compute_mean_width(lower, upper)
    score = compute_ci_score(cov, w, alpha, lambda_=lambda_, epsilon=epsilon)
    return {"coverage": cov, "width": w, "ci_score": score}
