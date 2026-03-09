"""Nonconformity score computation and mu-hat calibration distributions.

Pure-function API: every function takes arrays / dicts and returns arrays
/ dicts.  File I/O helpers are separate from computation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from conformal_predictions.training.core import (
    _compute_mu_hat,
)

# ---------------------------------------------------------------------------
# Elementary score function
# ---------------------------------------------------------------------------


def _random_perturbation() -> float:
    """Tiny noise for numerical stability (avoids ties in quantiles)."""
    return np.random.normal(0, 1e-6)


def nonconformity_score(pred: float, target: float, how: str = "diff") -> float:
    """Compute a single nonconformity score.

    Parameters
    ----------
    pred : float
        Predicted value.
    target : float
        True (reference) value.
    how : str
        ``"diff"`` → ``target - pred``;  ``"abs"`` → ``|target - pred|``.

    Returns
    -------
    float
    """
    if how == "diff":
        score = target - pred
    elif how == "abs":
        score = abs(target - pred)
    else:
        raise ValueError(f"Unknown score method: {how!r} (expected 'diff' or 'abs')")
    return score + _random_perturbation()


# ---------------------------------------------------------------------------
# Batch score computation
# ---------------------------------------------------------------------------


def compute_nonconformity_scores(
    models: Dict[str, object],
    scaler: StandardScaler,
    calib_data: Sequence[Tuple[np.ndarray, np.ndarray]],
    calib_meta: Sequence[dict],
    threshold: float,
    target: str = "mu_hat",
    how: str = "diff",
    ref_efficiencies: Optional[Sequence[float]] = None,
) -> Dict[str, List[float]]:
    """Compute nonconformity scores on calibration pseudo-experiments.

    Parameters
    ----------
    target : str
        ``"mu_hat"`` for signal-strength scores, ``"n_pred"`` for event count.
    how : str
        ``"diff"`` or ``"abs"``.
    ref_efficiencies : sequence of float, optional
        ``(eps_signal, eps_background)`` per model.  Required when
        ``target == "mu_hat"``.

    Returns
    -------
    Dict mapping model name → list of scores (one per experiment).
    """
    scores: Dict[str, List[float]] = {name: [] for name in models}

    for (X_calib, y_calib), _meta in tqdm(
        zip(calib_data, calib_meta),
        total=len(calib_data),
        desc="Computing nonconformity scores",
    ):
        X_scaled = scaler.transform(X_calib)
        for name, model in models.items():
            y_proba = model.predict_proba(X_scaled)[:, 1]
            n_pred = int(np.sum(y_proba > threshold))

            if target == "mu_hat":
                mu_true = _meta["mu_true"]
                mu_hat = _compute_mu_hat(n_pred, _meta, ref_efficiencies)
                scores[name].append(nonconformity_score(mu_hat, mu_true, how=how))
            elif target == "n_pred":
                n_obs = int(np.sum(y_calib))
                scores[name].append(nonconformity_score(n_pred, n_obs, how=how))
            else:
                raise ValueError(
                    f"Unknown target: {target!r} (expected 'mu_hat' or 'n_pred')"
                )

    return scores


# ---------------------------------------------------------------------------
# mu-hat calibration distribution
# ---------------------------------------------------------------------------


def compute_mu_hat(
    models: Dict[str, object],
    scaler: StandardScaler,
    calib_data: Sequence[Tuple[np.ndarray, np.ndarray]],
    calib_meta: Sequence[dict],
    threshold: float,
    ref_efficiencies: Sequence[float] = (1.0, 1.0),
) -> Tuple[Dict[str, List[float]], Dict[str, Dict[str, float]]]:
    """Compute mu-hat distribution on the calibration set.

    Returns
    -------
    mu_hat : dict
        model_name → list of mu-hat values.
    stats : dict
        model_name → summary statistics dict.
    """
    mu_hat: Dict[str, List[float]] = {name: [] for name in models}

    for (X_calib, _y_calib), meta in zip(calib_data, calib_meta):
        X_scaled = scaler.transform(X_calib)
        if meta["gamma_true"] == 0:
            continue
        for name, model in models.items():
            y_proba = model.predict_proba(X_scaled)[:, 1]
            n_pred = int(np.sum(y_proba > threshold))
            mu_pred = _compute_mu_hat(n_pred, meta, ref_efficiencies)
            mu_hat[name].append(mu_pred)

    stats: Dict[str, Dict[str, float]] = {}
    for name, values in mu_hat.items():
        if len(values) > 0:
            density = gaussian_kde(values)
            xs = np.linspace(min(values), max(values), 1000)
            density_vals = density(xs)
            map_estimate = float(xs[np.argmax(density_vals)])
            stats[name] = {
                "q16": float(np.percentile(values, 16)),
                "map": map_estimate,
                "mu_median": float(np.median(values)),
                "mu_mean": float(np.mean(values)),
                "q68": float(np.percentile(values, 68)),
                "q84": float(np.percentile(values, 84)),
            }

    return mu_hat, stats


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def save_scores(
    scores: Dict[str, List[float]],
    output_dir: Path,
    *,
    filename: str = "nonconformity_scores.npz",
    save_distribution_csv: bool = True,
) -> Path:
    """Persist nonconformity scores as ``.npz`` and optionally a histogram CSV.

    Returns the path to the saved ``.npz`` file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / filename
    np.savez(npz_path, **{n: np.array(v) for n, v in scores.items()})

    if save_distribution_csv:
        rows = []
        for name, vals in scores.items():
            arr = np.array(vals)
            if len(arr) == 0:
                continue
            counts, bin_edges = np.histogram(arr, bins=50)
            for i in range(len(counts)):
                rows.append(
                    {
                        "model": name,
                        "bin_low": float(bin_edges[i]),
                        "bin_high": float(bin_edges[i + 1]),
                        "count": int(counts[i]),
                    }
                )
        if rows:
            csv_path = output_dir / filename.replace(".npz", "_distribution.csv")
            pd.DataFrame(rows).to_csv(csv_path, index=False)

    return npz_path
