"""
Toy data generation utilities.

This module defines simple, fully controlled toy datasets
to validate conformal prediction methods and signal strength
aggregation strategies.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np


# ----------------------------------------------------------------------
# Configuration dataclass
# ----------------------------------------------------------------------


@dataclass
class ToyConfig:
    """
    Configuration for toy dataset generation.
    """

    n_samples: int
    n_features: int
    mu: float  # signal strength
    signal_fraction_ref: float  # reference signal fraction (mu = 1)
    random_state: Optional[int] = None


# ----------------------------------------------------------------------
# Main public API
# ----------------------------------------------------------------------


def generate_toy_dataset(
    cfg: ToyConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate a toy binary classification dataset with a controllable
    signal strength parameter mu.

    Parameters
    ----------
    cfg : ToyConfig
        Configuration object defining dataset properties.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray, shape (n_samples,)
        Binary labels (0 = background, 1 = signal).
    meta : dict
        Metadata dictionary containing ground-truth information
        (e.g. true mu, class counts).
    """
    rng = np.random.default_rng(cfg.random_state)

    # ------------------------------------------------------------------
    # Determine number of signal/background events
    # ------------------------------------------------------------------
    n_signal_ref = int(cfg.signal_fraction_ref * cfg.n_samples)
    n_signal = rng.poisson(cfg.mu * n_signal_ref)
    n_signal = min(n_signal, cfg.n_samples)
    n_background = cfg.n_samples - n_signal

    # ------------------------------------------------------------------
    # Generate features
    # (simple Gaussian example; can be refined later)
    # ------------------------------------------------------------------
    X_signal = rng.normal(
        loc=1.0,
        scale=1.0,
        size=(n_signal, cfg.n_features),
    )
    X_background = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(n_background, cfg.n_features),
    )

    X = np.vstack([X_signal, X_background])
    y = np.concatenate(
        [
            np.ones(n_signal, dtype=int),
            np.zeros(n_background, dtype=int),
        ]
    )

    # ------------------------------------------------------------------
    # Shuffle dataset
    # ------------------------------------------------------------------
    perm = rng.permutation(cfg.n_samples)
    X = X[perm]
    y = y[perm]

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    meta = {
        "mu_true": cfg.mu,
        "n_signal": n_signal,
        "n_background": n_background,
        "n_samples": cfg.n_samples,
    }

    return X, y, meta


# ----------------------------------------------------------------------
# Optional helper (explicitly separated)
# ----------------------------------------------------------------------


def make_default_toy_config(
    random_state: Optional[int] = None,
) -> ToyConfig:
    """
    Convenience function returning a reasonable default ToyConfig.
    """
    return ToyConfig(
        n_samples=10_000,
        n_features=2,
        mu=1.0,
        signal_fraction_ref=0.1,
        random_state=random_state,
    )
