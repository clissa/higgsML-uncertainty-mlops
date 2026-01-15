"""
Toy data generation utilities.

This module defines simple, fully controlled toy datasets
to validate conformal prediction methods and signal strength
aggregation strategies.
"""

from __future__ import annotations

import uuid
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml  # PyYAML

# ----------------------------------------------------------------------
# Configuration dataclass
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ToyConfig:
    """
    Toy data generator config.

    Core physics-inspired model:
        nu = mu * gamma + beta
        N_signal ~ Poisson(mu * gamma)
        N_background ~ Poisson(beta)

    Once n_signal and n_background are drawn, features are sampled from
    class-conditional multivariate normals. This generates one pseudo-experiment.

    Attributes signal_weight and background_weight define per-sample scale factors
    assigned to each class.

    Notes on reproducibility:
    - seed is deterministic by default.
    - set seed=None for statistical (non-deterministic) runs.
    """

    # Physics / counting parameters
    mu: float  # signal strength
    gamma: float  # expected signal events
    beta: float  # expected background events

    # Feature space
    n_features: int = 2

    # Signal feature distribution parameters
    signal_mean: np.ndarray[np.float64]  # shape: (n_features,)
    signal_std: np.ndarray[np.float64]  # shape: (n_features,)
    signal_rho: np.ndarray[np.float64]  # shape: (n_features-1,)

    # Background feature distribution parameters
    background_mean: np.ndarray[np.float64]  # shape: (n_features,)
    background_std: np.ndarray[np.float64]  # shape: (n_features,)
    background_rho: np.ndarray[np.float64]  # shape: (n_features-1,)

    # Sample weights
    signal_weight: float
    background_weight: float

    # Random seed (None → statistical reproducibility only)
    seed: Optional[int] = 4


# ----------------------------------------------------------------------
# Main public API
# ----------------------------------------------------------------------

## Helper functions


def _validate_pseudo_experiment_inputs(
    config: ToyConfig, pseudo_experiment_id: str
) -> None:
    # pseudo_experiment_id checks: type, length, hex
    if not isinstance(pseudo_experiment_id, str):
        raise ValueError("pseudo_experiment_id must be a HEX string of 16 chars")
    if len(pseudo_experiment_id) != 16:
        raise ValueError("pseudo_experiment_id must be a HEX string of 16 chars")
    try:
        int(pseudo_experiment_id, 16)
    except ValueError as exc:
        raise ValueError(
            "pseudo_experiment_id must be a HEX string of 16 chars"
        ) from exc

    # physics / counting parameters
    if config.mu < 0:
        raise ValueError("mu must be >= 0")
    if config.gamma < 0:
        raise ValueError("gamma must be >= 0")
    if config.beta < 0:
        raise ValueError("beta must be >= 0")
    if config.n_features <= 0:
        raise ValueError("n_features must be > 0")

    expected_vector_shape = (config.n_features,)
    expected_rho_shape = (config.n_features - 1,)

    # signal shapes
    signal_mean = np.asarray(config.signal_mean)
    signal_std = np.asarray(config.signal_std)
    signal_rho = np.asarray(config.signal_rho)

    if signal_mean.shape != expected_vector_shape:
        raise ValueError(f"signal_mean must have shape {expected_vector_shape}")
    if signal_std.shape != expected_vector_shape:
        raise ValueError(f"signal_std must have shape {expected_vector_shape}")
    if signal_rho.shape != expected_rho_shape:
        raise ValueError(f"signal_rho must have shape {expected_rho_shape}")

    # background shapes
    background_mean = np.asarray(config.background_mean)
    background_std = np.asarray(config.background_std)
    background_rho = np.asarray(config.background_rho)

    if background_mean.shape != expected_vector_shape:
        raise ValueError(f"background_mean must have shape {expected_vector_shape}")
    if background_std.shape != expected_vector_shape:
        raise ValueError(f"background_std must have shape {expected_vector_shape}")
    if background_rho.shape != expected_rho_shape:
        raise ValueError(f"background_rho must have shape {expected_rho_shape}")

    # std checks
    if np.any(signal_std < 0):
        raise ValueError("signal_std must be >= 0 for all features")
    if np.any(background_std < 0):
        raise ValueError("background_std must be >= 0 for all features")

    # rho checks
    if np.any((signal_rho < -1) | (signal_rho > 1)):
        raise ValueError("signal_rho values must be in [-1, 1]")
    if np.any((background_rho < -1) | (background_rho > 1)):
        raise ValueError("background_rho values must be in [-1, 1]")


def _covariance_from_std_and_adjacent_rho(
    standard_deviations: np.ndarray[np.float64],
    adjacent_correlations: np.ndarray[np.float64],
) -> np.ndarray[np.float64]:
    n_features = int(standard_deviations.shape[0])

    if adjacent_correlations.shape != (n_features - 1,):
        raise ValueError(
            f"adjacent_correlations must have shape ({n_features - 1},), "
            f"got {adjacent_correlations.shape}"
        )

    if np.any(standard_deviations < 0):
        raise ValueError("standard_deviations must be >= 0 for all features")

    covariance = np.zeros((n_features, n_features), dtype=np.float64)
    np.fill_diagonal(covariance, standard_deviations * standard_deviations)

    for feature_index in range(n_features - 1):
        rho_value = float(adjacent_correlations[feature_index])
        if not (-1.0 <= rho_value <= 1.0):
            raise ValueError(
                f"Correlation rho[{feature_index}] must be in [-1, 1], got {rho_value}"
            )

        covariance_value = (
            rho_value
            * float(standard_deviations[feature_index])
            * float(standard_deviations[feature_index + 1])
        )
        covariance[feature_index, feature_index + 1] = covariance_value
        covariance[feature_index + 1, feature_index] = covariance_value

    return covariance


def generate_pseudo_experiment(
    cfg: ToyConfig,
    pseudo_experiment_id: Optional[str] = None,
) -> Tuple[np.ndarray[np.float32], np.ndarray[np.int64], Dict[str, Any]]:
    """
    Generate a single pseudo-experiment according to the ToyConfig model.

    pseudo_experiment_id: str    Hex string identifier (e.g. first 16 chars of UUID).

    Steps:
      1) Draw counts:
         - n_signal ~ Poisson(mu * gamma)
         - n_background ~ Poisson(beta)
      2) Sample features from class-conditional multivariate normals.
      3) Concatenate into one dataset and return (X, y, meta).

    Returns:
        X: shape (n_total, n_features), dtype float32
        y: shape (n_total,), dtype int64 (1=signal, 0=background)
        meta: dictionary containing:
            - pseudo_experiment_id
            - mu_true, gamma_true, beta_true, nu_expected
            - n_signal, n_background, n_total
            - weights: per-sample weights
            - feature_params: means/std/rho used (float64)
            - seed used (base seed, can be None)
    """
    if pseudo_experiment_id is None:
        pseudo_experiment_id = uuid.uuid4().hex[:16]
    _validate_pseudo_experiment_inputs(cfg, pseudo_experiment_id)

    # set seed for all generations in this pseudo-experiment
    if cfg.seed is None:
        random_generator = np.random.default_rng()
    else:
        experiment_seed = (cfg.seed + int(pseudo_experiment_id, 16)) % (2**32)
        random_generator = np.random.default_rng(experiment_seed)

    n_signal = random_generator.poisson(lam=cfg.mu * cfg.gamma)
    n_background = random_generator.poisson(lam=cfg.beta)
    n_total = n_signal + n_background

    if n_total == 0:
        empty_features = np.empty((0, cfg.n_features), dtype=np.float32)
        empty_labels = np.empty((0,), dtype=np.int64)
        empty_weights = np.empty((0,), dtype=np.float64)

        meta: Dict[str, Any] = {
            "pseudo_experiment_id": pseudo_experiment_id,
            "mu_true": float(cfg.mu),
            "gamma_true": float(cfg.gamma),
            "beta_true": float(cfg.beta),
            "nu_expected": float(cfg.mu * cfg.gamma + cfg.beta),
            "n_signal": 0,
            "n_background": 0,
            "n_total": 0,
            "weights": empty_weights,
            "seed": cfg.seed,
        }
        return empty_features, empty_labels, meta

    signal_mean = np.asarray(cfg.signal_mean, dtype=np.float64)
    signal_standard_deviations = np.asarray(cfg.signal_std, dtype=np.float64)
    signal_correlations = np.asarray(cfg.signal_rho, dtype=np.float64)
    signal_covariance = _covariance_from_std_and_adjacent_rho(
        standard_deviations=signal_standard_deviations,
        adjacent_correlations=signal_correlations,
    )

    background_mean = np.asarray(cfg.background_mean, dtype=np.float64)
    background_standard_deviations = np.asarray(cfg.background_std, dtype=np.float64)
    background_correlations = np.asarray(cfg.background_rho, dtype=np.float64)
    background_covariance = _covariance_from_std_and_adjacent_rho(
        standard_deviations=background_standard_deviations,
        adjacent_correlations=background_correlations,
    )

    feature_blocks: list[np.ndarray[np.float32]] = []
    label_blocks: list[np.ndarray[np.int64]] = []
    weight_blocks: list[np.ndarray[np.float32]] = []

    if n_signal > 0:
        signal_features = random_generator.multivariate_normal(
            mean=signal_mean,
            cov=signal_covariance,
            size=n_signal,
        ).astype(np.float32, copy=False)

        signal_labels = np.ones((n_signal,), dtype=np.int64)
        signal_weights = np.full(
            (n_signal,),
            fill_value=float(cfg.signal_weight / n_signal),
            dtype=np.float32,
        )

        feature_blocks.append(signal_features)
        label_blocks.append(signal_labels)
        weight_blocks.append(signal_weights)

    if n_background > 0:
        background_features = random_generator.multivariate_normal(
            mean=background_mean,
            cov=background_covariance,
            size=n_background,
        ).astype(np.float32, copy=False)

        background_labels = np.zeros((n_background,), dtype=np.int64)
        background_weights = np.full(
            (n_background,),
            fill_value=float(cfg.background_weight / n_background),
            dtype=np.float32,
        )

        feature_blocks.append(background_features)
        label_blocks.append(background_labels)
        weight_blocks.append(background_weights)

    X = np.vstack(feature_blocks).astype(np.float32, copy=False)
    y = np.concatenate(label_blocks).astype(np.int64, copy=False)
    weights = np.concatenate(weight_blocks).astype(np.float32, copy=False)

    shuffle_indices = random_generator.permutation(n_total)
    X = X[shuffle_indices]
    y = y[shuffle_indices]
    weights = weights[shuffle_indices]

    meta = {
        "pseudo_experiment_id": pseudo_experiment_id,
        "mu_true": float(cfg.mu),
        "gamma_true": float(cfg.gamma),
        "beta_true": float(cfg.beta),
        "nu_expected": float(cfg.mu * cfg.gamma + cfg.beta),
        "n_signal": int(n_signal),
        "n_background": int(n_background),
        "n_total": int(n_total),
        "weights": weights,
        "feature_params": {
            "n_features": int(cfg.n_features),
            "signal": {
                "mean": signal_mean,
                "std": signal_standard_deviations,
                "rho": signal_correlations,
                "covariance": signal_covariance,
            },
            "background": {
                "mean": background_mean,
                "std": background_standard_deviations,
                "rho": background_correlations,
                "covariance": background_covariance,
            },
        },
        "seed": cfg.seed,
    }

    return X, y, meta


def load_toy_config_from_yaml(yaml_path: str | Path) -> ToyConfig:
    """
    Read a YAML config file and build a ToyConfig instance.

    Tasks performed:
      - loads YAML into a dict
      - checks required fields from ToyConfig dataclass
      - converts list-like feature params into numpy arrays with expected dtypes
      - errors on unknown fields (to catch typos early)
    """
    yaml_path = Path(yaml_path)
    raw: Dict[str, Any] = yaml.safe_load(yaml_path.read_text())

    if not isinstance(raw, dict):
        raise ValueError("YAML root must be a mapping (dict-like).")

    field_names = {f.name for f in fields(ToyConfig)}
    required_names = {
        f.name
        for f in fields(ToyConfig)
        if f.default is MISSING and f.default_factory is MISSING
    }

    missing = sorted(required_names - raw.keys())
    if missing:
        raise ValueError(f"Missing required ToyConfig fields in YAML: {missing}")

    unknown = sorted(set(raw.keys()) - field_names)
    if unknown:
        raise ValueError(f"Unknown fields in YAML (typos?): {unknown}")

    # Convert array-like fields
    def _as_f16_array(value: Any, name: str) -> np.ndarray:
        try:
            return np.asarray(value, dtype=np.float16)
        except Exception as exc:
            raise ValueError(
                f"Field '{name}' must be array-like (list/sequence)."
            ) from exc

    for name in (
        "signal_mean",
        "signal_std",
        "signal_rho",
        "background_mean",
        "background_std",
        "background_rho",
    ):
        if name in raw:
            raw[name] = _as_f16_array(raw[name], name)

    # Construct dataclass (let _validate_pseudo_experiment_inputs do shape/range checks)
    cfg = ToyConfig(**raw)  # type: ignore[arg-type]
    return cfg


def generate_pseudo_experiment_from_yaml(
    yaml_path: str | Path,
    pseudo_experiment_id: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Convenience wrapper:
      1) read YAML -> ToyConfig
      2) generate one pseudo-experiment with existing generator
    """
    cfg = load_toy_config_from_yaml(yaml_path)
    return generate_pseudo_experiment(
        cfg=cfg, pseudo_experiment_id=pseudo_experiment_id
    )
