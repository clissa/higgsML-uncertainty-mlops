"""Model factory for the conformal-prediction pipeline.

Builds a **single** sklearn estimator from a :class:`ModelConfig`.
Supported model families: MLP (default), GLM (logistic regression),
Random Forest.

Usage::

    from conformal_predictions.config import ModelConfig
    from conformal_predictions.training.models import build_model

    models = build_model(ModelConfig(name="mlp"), seed=42)
    # models == {"MLP": MLPClassifier(...)}
"""

from __future__ import annotations

import warnings
from typing import Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from conformal_predictions.config import ModelConfig

# Display name for each model family
_DISPLAY_NAMES = {
    "mlp": "MLP",
    "glm": "GLM",
    "random_forest": "Random Forest",
}

# Default hyperparameters per model family (overridable via ModelConfig.params)
_DEFAULTS: Dict[str, dict] = {
    "mlp": {
        "hidden_layer_sizes": (32, 16),
        "activation": "relu",
        "max_iter": 1000,
    },
    "glm": {
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 1000,
    },
    "random_forest": {
        "n_estimators": 50,
        "criterion": "gini",
        "n_jobs": 1,
    },
}

_CLASSES = {
    "mlp": MLPClassifier,
    "glm": LogisticRegression,
    "random_forest": RandomForestClassifier,
}


def build_model(
    model_config: ModelConfig,
    seed: int,
) -> Dict[str, object]:
    """Instantiate a single model from *model_config*.

    Returns a ``{display_name: estimator}`` dict (single entry) to keep
    downstream code — which already iterates ``dict.items()`` — working
    unchanged.

    Parameters
    ----------
    model_config : ModelConfig
        Model family and optional hyperparameter overrides.
    seed : int
        Random state for reproducibility.
    """
    name = model_config.name
    cls = _CLASSES[name]
    display = _DISPLAY_NAMES[name]

    # Merge: defaults ← user params ← seed
    kwargs = {**_DEFAULTS[name], **model_config.params, "random_state": seed}

    # Convert list → tuple for hidden_layer_sizes (YAML parses as list)
    if "hidden_layer_sizes" in kwargs and isinstance(
        kwargs["hidden_layer_sizes"], list
    ):
        kwargs["hidden_layer_sizes"] = tuple(kwargs["hidden_layer_sizes"])

    return {display: cls(**kwargs)}


# ---------------------------------------------------------------------------
# Backward-compatible alias (deprecated)
# ---------------------------------------------------------------------------


def build_default_models(seed: int, n_jobs: int = 1) -> Dict[str, object]:
    """Return the legacy three-model catalogue.

    .. deprecated::
        Use :func:`build_model` with a :class:`ModelConfig` instead.
        This function is kept only for backward-compatible legacy scripts.
    """
    warnings.warn(
        "build_default_models() is deprecated. "
        "Use build_model(ModelConfig(...), seed) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
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
