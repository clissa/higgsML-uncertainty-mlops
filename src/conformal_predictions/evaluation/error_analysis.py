"""Error analysis utilities for the conformal-prediction pipeline.

Provides ``build_top_errors_table`` which ranks examples by per-example
loss and returns a ``pandas.DataFrame`` (and optionally a ``wandb.Table``).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from conformal_predictions.evaluation.metrics import compute_per_example_loss


def build_top_errors_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    X: Optional[np.ndarray] = None,
    feature_names: Optional[Sequence[str]] = None,
    N: int = 50,
) -> pd.DataFrame:
    """Build a table of the *N* worst-predicted examples.

    Parameters
    ----------
    y_true : 1-D array
        Ground-truth labels.
    y_pred : 1-D array
        Hard predictions (0/1).
    y_proba : 1-D array
        Predicted probabilities for the positive class.
    X : 2-D array, optional
        Feature matrix; columns are included in the output table.
    feature_names : sequence of str, optional
        Column names for *X*.  Defaults to ``feat_0, feat_1, …``.
    N : int
        Number of top-error examples to return.

    Returns
    -------
    pd.DataFrame
        Sorted by ``per_example_loss`` descending, containing columns:
        ``index``, ``true_label``, ``predicted_label``, ``prediction_score``,
        ``per_example_loss``, ``confidence``, ``confidence_margin``, and
        optionally feature columns.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba, dtype=float)

    losses = compute_per_example_loss(y_true, y_proba)
    confidence = np.where(y_pred == 1, y_proba, 1.0 - y_proba)
    confidence_margin = np.abs(y_proba - 0.5)

    df = pd.DataFrame(
        {
            "index": np.arange(len(y_true)),
            "true_label": y_true.astype(int),
            "predicted_label": y_pred.astype(int),
            "prediction_score": y_proba,
            "per_example_loss": losses,
            "confidence": confidence,
            "confidence_margin": confidence_margin,
        }
    )

    if X is not None:
        X_arr = np.asarray(X)
        if feature_names is None:
            feature_names = [f"feat_{i}" for i in range(X_arr.shape[1])]
        for i, fname in enumerate(feature_names):
            df[fname] = X_arr[:, i]

    df = (
        df.sort_values("per_example_loss", ascending=False)
        .head(N)
        .reset_index(drop=True)
    )
    return df


def build_top_errors_wandb_table(
    df: pd.DataFrame,
) -> object:
    """Convert a top-errors DataFrame into a ``wandb.Table``.

    Returns *None* when wandb is unavailable.
    """
    try:
        import wandb  # type: ignore[import]

        return wandb.Table(dataframe=df)
    except Exception:
        return None
