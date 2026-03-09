"""Default model definitions for the conformal-prediction pipeline.

Centralises the model catalogue so that both the toy and Higgs training
scripts share a single source of truth.

TODO Phase 5: Make model specs config-driven (model type + hyperparams
in YAML) for sweep support.
"""

from __future__ import annotations

from typing import Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def build_default_models(seed: int, n_jobs: int = 1) -> Dict[str, object]:
    """Return the standard model catalogue.

    Parameters
    ----------
    seed : int
        Random state for reproducibility.
    n_jobs : int, optional
        Number of parallel jobs for models that support it (default 1).
    """
    return {
        "GLM": LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            random_state=seed,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=50,
            criterion="gini",
            n_jobs=n_jobs,
            random_state=seed,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            max_iter=1000,
            random_state=seed,
        ),
    }
