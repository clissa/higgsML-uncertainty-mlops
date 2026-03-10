"""Model factory for the conformal-prediction pipeline.

Builds a **single** MLP estimator from a :class:`ModelConfig`.

This pipeline follows a single-model-per-run design: one config, one
model, one run.  Only ``"mlp"`` is supported in the main path.

Usage::

    from conformal_predictions.config import ModelConfig
    from conformal_predictions.training.models import build_model

    models = build_model(ModelConfig(name="mlp"), seed=42)
    # models == {"MLP": MLPClassifier(...)}
"""

from __future__ import annotations

import warnings
from typing import Dict

from sklearn.neural_network import MLPClassifier

from conformal_predictions.config import ModelConfig

# Display name for MLP
_DISPLAY_NAME = "MLP"

# Default hyperparameters (overridable via ModelConfig.params)
_DEFAULTS: dict = {
    "hidden_layer_sizes": (32, 16),
    "activation": "relu",
    "max_iter": 1000,
}


def build_model(
    model_config: ModelConfig,
    seed: int,
) -> Dict[str, object]:
    """Instantiate a single MLP from *model_config*.

    Returns a ``{"MLP": estimator}`` dict so that downstream code that
    iterates ``dict.items()`` continues to work unchanged.

    Parameters
    ----------
    model_config : ModelConfig
        Model config.  Only ``name="mlp"`` is supported.
    seed : int
        Random state for reproducibility.
    """
    # ModelConfig.__post_init__ already guards against non-mlp names.
    # Merge: defaults ← user params ← seed
    kwargs = {**_DEFAULTS, **model_config.params, "random_state": seed}

    # Convert list → tuple for hidden_layer_sizes (YAML parses as list)
    if "hidden_layer_sizes" in kwargs and isinstance(
        kwargs["hidden_layer_sizes"], list
    ):
        kwargs["hidden_layer_sizes"] = tuple(kwargs["hidden_layer_sizes"])

    return {_DISPLAY_NAME: MLPClassifier(**kwargs)}


# ---------------------------------------------------------------------------
# Backward-compatible alias (deprecated)
# ---------------------------------------------------------------------------


def build_default_models(seed: int, n_jobs: int = 1) -> Dict[str, object]:
    """Return the legacy three-model catalogue.

    .. deprecated::
        Use :func:`build_model` with a :class:`ModelConfig` instead.
        This function is kept only for backward-compatible legacy scripts.
        The main pipeline no longer trains multiple models per run.
    """
    warnings.warn(
        "build_default_models() is deprecated and will be removed in a future "
        "version.  Use build_model(ModelConfig(name='mlp'), seed) for the "
        "single-model pipeline.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Lazy imports — GLM/RF are not part of the main pipeline any more.
    from sklearn.ensemble import RandomForestClassifier  # noqa: PLC0415
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

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
