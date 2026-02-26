from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import gaussian_kde
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


def _experiment_prefix(path: Path) -> str:
    stem = path.stem
    if stem.startswith("experiment_"):
        stem = stem[len("experiment_") :]
    return stem[:4]


def list_split_files(
    data_dir: Path,
    mu: float,
    test_prefixes: Optional[Sequence[str]],
    n_test_experiments: Optional[int],
    valid_size: float,
    calib_size: float,
    seed: int,
) -> Tuple[List[Path], List[Path], List[Path], List[Path]]:
    if valid_size < 0 or calib_size < 0 or valid_size + calib_size >= 1:
        raise ValueError("valid_size and calib_size must be >= 0 and sum to < 1.")
    mu_dir = data_dir / f"mu={mu}"
    files = sorted(mu_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {mu_dir}")
    rng = np.random.default_rng(seed)
    test_files: List[Path] = []
    if test_prefixes:
        test_files = [
            path for path in files if _experiment_prefix(path) in test_prefixes
        ]
    if not test_files:
        print(
            "No test files found with the specified prefixes. Falling back to random files."
        )
        if not n_test_experiments:
            raise ValueError("n_test_experiments must be set when no prefixes match.")
        n_test = min(n_test_experiments, len(files))
        test_files = list(rng.choice(files, size=n_test, replace=False))
    remaining_files = [path for path in files if path not in test_files]
    if not remaining_files:
        raise ValueError("No files remain after test split.")
    n_calib = int(np.floor(calib_size * len(remaining_files)))
    calib_files: List[Path] = []
    if n_calib > 0:
        calib_files = list(rng.choice(remaining_files, size=n_calib, replace=False))
    train_val_files = [path for path in remaining_files if path not in calib_files]
    if not train_val_files:
        raise ValueError("No files remain after calibration split.")
    n_val = int(np.floor(valid_size * len(train_val_files)))
    val_files: List[Path] = []
    if n_val > 0:
        val_files = list(rng.choice(train_val_files, size=n_val, replace=False))
    train_files = [path for path in train_val_files if path not in val_files]
    if not train_files:
        raise ValueError("No training files remain after train/val split.")
    return train_files, val_files, calib_files, test_files


def evaluate_models(
    models: Dict[str, object], X: np.ndarray, y: np.ndarray
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for name, model in models.items():
        y_pred = model.predict(X)
        results[name] = {
            "accuracy": float(model.score(X, y)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
        }
    return results


def get_events_count(
    models: Dict[str, object], X: np.ndarray, threshold: float
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X)[:, 1]
        counts[name] = int(np.sum(y_pred_proba > threshold))
    return counts


def _random_perturbation_for_numerical_stability() -> float:
    return np.random.normal(0, 1e-6)


def _nonconformity_scores(pred, target, how: str = "diff") -> float:
    """Compute nonconformity score based on the difference between prediction and target.
    Args: how: diff (target - pred) or abs_diff (|target - pred|)"""
    if how == "diff":
        score = target - pred
    elif how == "abs":
        score = abs(target - pred)
    else:
        raise ValueError(f"Unknown how value: {how}")
    return score + _random_perturbation_for_numerical_stability()


def _get_proportionate_gamma(meta: dict) -> float:
    return meta["gamma_true"] / meta["nu_expected"] * meta["n_total"]


def compute_nonconformity_scores(
    models: Dict[str, object],
    scaler: StandardScaler,
    calib_data: Sequence[Tuple[np.ndarray, np.ndarray]],
    calib_meta: Sequence[dict],
    threshold: float,
    target: str = "mu_hat",  # can be "n_pred" or "mu_hat",
    how: str = "diff",  # method for computing nonconformity scores: "diff" or "abs"
) -> Dict[str, List[int]]:
    scores: Dict[str, List[int]] = {name: [] for name in models}
    for (X_calib, y_calib), _meta in tqdm(
        zip(calib_data, calib_meta),
        total=len(calib_data),
        desc="Computing nonconformity scores",
    ):

        X_calib = scaler.transform(X_calib)
        n_obs = int(np.sum(y_calib))
        mu_true = _meta["mu_true"]
        gamma_true = _get_proportionate_gamma(_meta)
        for name, model in models.items():
            y_pred_proba = model.predict_proba(X_calib)[:, 1]
            n_pred = int(np.sum(y_pred_proba > threshold))
            if target == "mu_hat":
                mu_hat = n_pred / gamma_true if gamma_true > 0 else 0.0
                scores[name].append(_nonconformity_scores(mu_hat, mu_true, how=how))
            elif target == "n_pred":
                scores[name].append(_nonconformity_scores(n_pred, n_obs, how=how))
    return scores


def compute_mu_hat(
    models: Dict[str, object],
    scaler: StandardScaler,
    calib_data: Sequence[Tuple[np.ndarray, np.ndarray]],
    calib_meta: Sequence[dict],
    threshold: float,
) -> Tuple[Dict[str, List[float]], Dict[str, Dict[str, float]]]:
    mu_hat: Dict[str, List[float]] = {name: [] for name in models}
    for (X_calib, y_calib), meta in zip(calib_data, calib_meta):
        X_calib = scaler.transform(X_calib)
        gamma_true = _get_proportionate_gamma(meta)
        if gamma_true == 0:
            continue
        for name, model in models.items():
            y_pred_proba = model.predict_proba(X_calib)[:, 1]
            n_pred = int(np.sum(y_pred_proba > threshold))
            mu_hat[name].append(n_pred / gamma_true)

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
                "q84": float(np.percentile(values, 84)),
            }

    return mu_hat, stats


def inference_on_test_set(
    models: Dict[str, object],
    scaler: StandardScaler,
    test_data: Sequence[Tuple[np.ndarray, np.ndarray, dict]],
    threshold: float,
    debug: bool = False,
) -> Tuple[Dict[str, List[float]], List[float], List[int]]:
    """
    Compute mu_hat estimates on test set pseudo-experiments.

    Args:
        models: Dictionary of trained models
        scaler: Fitted StandardScaler for feature normalization
        test_data: Sequence of (X, y, meta_dict) tuples for test experiments
        threshold: Decision threshold for classification

    Returns:
        mu_hat_test: Dictionary mapping model names to lists of mu_hat values (one per experiment)
        mu_true_list: List of mu_true values (one per experiment)
        gamma_true_list: List of gamma_true values (one per experiment)
    """
    mu_hat_test: Dict[str, List[float]] = {name: [] for name in models}
    performance_metrics: Dict[str, List[Dict[str, float]]] = {
        name: [] for name in models
    }
    mu_true_list: List[float] = []
    gamma_true_list: List[int] = []

    for X_test, y_test, meta_dict in tqdm(test_data, desc="Inference on test set"):
        X_test_scaled = scaler.transform(X_test)

        gamma_true = _get_proportionate_gamma(meta_dict)
        mu_true = meta_dict["mu_true"]
        mu_true_list.append(float(mu_true))
        gamma_true_list.append(int(gamma_true))

        if gamma_true == 0:
            continue

        for name, model in models.items():
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            y_pred = y_pred_proba > threshold

            # classification metrics
            _metrics = {
                "accuracy": float(model.score(X_test_scaled, y_test)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            }
            performance_metrics[name].append(_metrics)

            # counting metrics
            n_pred = int(np.sum(y_pred))

            mu_hat = n_pred / gamma_true
            mu_hat_test[name].append(mu_hat)

            if debug:
                print("\nDebug prints:", name)
                n_obs = int(np.sum(y_test))
                print(
                    f"\tExperiment: mu_true={mu_true:.4f}, gamma_true={gamma_true}, n_obs={n_obs}, n_pred={n_pred}, mu_hat={mu_hat:.4f}"
                )

    return mu_hat_test, mu_true_list, gamma_true_list, performance_metrics


def compute_confidence_interval(
    y_pred,
    nonconf_scores_file: Path,
    model_name: str,
    how: str = "diff",
) -> Tuple[float, float]:
    """
    Compute confidence interval from calibration nonconformity scores.

    Args:
        y_pred: Predicted value for which to compute the confidence interval
        nonconf_scores_file: Path to .npz file containing nonconformity scores
        model_name: Name of the model to extract scores for
        how: Method used to get nonconformity scores ("diff" or "abs")
    Returns:
        Tuple of (lower_bound, upper_bound) for the confidence interval
    """
    data = np.load(nonconf_scores_file)
    if model_name not in data:
        raise KeyError(f"Model '{model_name}' not found in {nonconf_scores_file}")

    scores = data[model_name]

    if how == "diff":
        q_low = float(np.percentile(scores, 16))
        q_high = float(np.percentile(scores, 84))
    elif how == "abs":
        q_low = -np.percentile(scores, 68)
        q_high = np.percentile(scores, 68)

    lower_bound = y_pred + q_low
    upper_bound = y_pred + q_high

    return lower_bound, upper_bound
