"""Configuration — YAML-loadable, frozen dataclasses.

Provides ``TrainingConfig``, ``CalibrationConfig``, and
``EvaluationConfig`` as composable, frozen configuration objects.

Usage::

    from conformal_predictions.config import load_training_config
    cfg = load_training_config("configs/train_toy.yaml")
    print(cfg.calibration.alpha)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional, Sequence, Tuple

import yaml

# ---------------------------------------------------------------------------
# Default: 1-sigma significance level  (alpha ≈ 0.3173)
# ---------------------------------------------------------------------------
ONE_SIGMA_ALPHA: float = 1.0 - math.erf(1.0 / math.sqrt(2.0))

# Default performance metric names
DEFAULT_METRICS: Tuple[str, ...] = (
    "loss",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "pr_auc",
    "roc_auc",
)


@dataclass(frozen=True)
class CalibrationConfig:
    """Configuration for the conformal calibration stage.

    Parameters
    ----------
    target : str
        Target variable for nonconformity score computation.
        ``"mu"`` for signal strength, ``"n_pred"`` for event count.
    how : str
        Score computation method: ``"diff"`` or ``"abs"``.
    alpha : float
        Significance level.  Default ≈ 0.3173 (1-sigma).
    ci_type : str
        ``"asymmetric"`` (two-sided quantiles) or ``"central"``
        (symmetric ± q(1-alpha)).
    enabled : bool
        Whether calibration runs as part of a pipeline.
    save_artifacts : bool
        Persist calibration scores, quantiles, and plots.
    """

    target: str = "mu"  # "mu" or "n_pred"
    how: str = "diff"  # "diff" or "abs"
    alpha: float = ONE_SIGMA_ALPHA
    ci_type: str = "asymmetric"  # "asymmetric" or "central"
    enabled: bool = True
    save_artifacts: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for the evaluation stage.

    Parameters
    ----------
    metrics : Sequence[str]
        Names of performance metrics to compute.
    enabled : bool
        Whether evaluation runs as part of a pipeline.
    save_artifacts : bool
        Persist metric outputs and plots.
    ci_score_lambda : float
        Coverage penalty coefficient for the CI score.
    ci_score_epsilon : float
        Small constant for numerical stability in CI score.
    """

    metrics: Sequence[str] = DEFAULT_METRICS
    enabled: bool = True
    save_artifacts: bool = True
    ci_score_lambda: float = 1.0
    ci_score_epsilon: float = 1e-6

    def to_dict(self) -> dict:
        d = asdict(self)
        # Ensure tuple serialises nicely
        d["metrics"] = list(d["metrics"])
        return d


@dataclass(frozen=True)
class TrainingConfig:
    """Frozen configuration for a training run.

    All fields have sensible defaults matching the toy pipeline so that
    ``TrainingConfig()`` is immediately usable for a quick local run.
    """

    # --- dataset identification ---
    dataset: str = "toy"  # "toy" or "higgs"
    data_dir: str = "data/toy_scale_easy"
    mu: float = 1.0

    # --- reproducibility ---
    seed: int = 18

    # --- data splitting (toy pipeline) ---
    test_prefixes: Sequence[str] = ("7e39", "6fcb")
    n_test_experiments: int = 1000
    valid_size: float = 0.2
    calib_size: float = 0.5

    # --- model / inference ---
    threshold: float = 0.5

    # Legacy fields — kept for backward compatibility.
    # New code should use ``calibration.target`` / ``calibration.how``.
    nonconf_target: str = "mu_hat"  # "mu_hat" or "n_pred"
    nonconf_method: str = "diff"  # "diff" or "abs"

    # --- output ---
    output_dir: str = "results"
    run_name: Optional[str] = None  # human-readable label; auto-generated if None

    # --- optional sub-configs (Phase 2) ---
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> dict:
        """Serialise to a plain dict (for JSON / run-context snapshots)."""
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Helpers for building CalibrationConfig from legacy fields
# ---------------------------------------------------------------------------

_TARGET_MAP = {"mu_hat": "mu", "n_pred": "n_pred"}


def _build_calibration_config(raw: dict) -> CalibrationConfig:
    """Build a ``CalibrationConfig`` from a YAML ``calibration:`` section.

    Falls back to legacy ``nonconf_target`` / ``nonconf_method`` when the
    dedicated section is absent.
    """
    calib_raw = raw.get("calibration")
    if calib_raw and isinstance(calib_raw, dict):
        valid_keys = {f.name for f in fields(CalibrationConfig)}
        filtered = {k: v for k, v in calib_raw.items() if k in valid_keys}
        return CalibrationConfig(**filtered)

    # Legacy fallback
    target = _TARGET_MAP.get(raw.get("nonconf_target", "mu_hat"), "mu")
    how = raw.get("nonconf_method", "diff")
    return CalibrationConfig(target=target, how=how)


def _build_evaluation_config(raw: dict) -> EvaluationConfig:
    """Build an ``EvaluationConfig`` from a YAML ``evaluation:`` section."""
    eval_raw = raw.get("evaluation")
    if eval_raw and isinstance(eval_raw, dict):
        valid_keys = {f.name for f in fields(EvaluationConfig)}
        filtered = {k: v for k, v in eval_raw.items() if k in valid_keys}
        if "metrics" in filtered and isinstance(filtered["metrics"], list):
            filtered["metrics"] = tuple(filtered["metrics"])
        return EvaluationConfig(**filtered)
    return EvaluationConfig()


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_training_config(path: str | Path) -> TrainingConfig:
    """Load a ``TrainingConfig`` from a YAML file.

    Unknown keys in the YAML are silently ignored so that config files
    can carry extra metadata without breaking the loader (forward-compat).

    Nested ``calibration:`` and ``evaluation:`` sections are parsed into
    their respective dataclasses.
    """
    path = Path(path)
    with open(path) as fh:
        raw: dict = yaml.safe_load(fh) or {}

    # Top-level scalar fields
    valid_keys = {f.name for f in fields(TrainingConfig)} - {
        "calibration",
        "evaluation",
    }
    filtered = {k: v for k, v in raw.items() if k in valid_keys}

    # Convert list → tuple for frozen dataclass compatibility
    if "test_prefixes" in filtered and isinstance(filtered["test_prefixes"], list):
        filtered["test_prefixes"] = tuple(filtered["test_prefixes"])

    # Sub-configs
    filtered["calibration"] = _build_calibration_config(raw)
    filtered["evaluation"] = _build_evaluation_config(raw)

    return TrainingConfig(**filtered)
