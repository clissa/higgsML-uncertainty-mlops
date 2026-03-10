"""Pseudo-experiment test inference and evaluation.

Decouples *inference* (running models on test experiments) from
*evaluation* (computing metrics on the results).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from conformal_predictions.mlops.run_context import RunContext

from conformal_predictions.calibration.intervals import compute_confidence_interval
from conformal_predictions.calibration.strategies import CalibrationResult
from conformal_predictions.config import CalibrationConfig, EvaluationConfig
from conformal_predictions.data_viz import plot_confidence_intervals
from conformal_predictions.evaluation.metrics import (
    compute_calibration_metrics,
)
from conformal_predictions.training.core import (
    _get_expected_background,
    _get_expected_signal,
    _get_proportionate_beta_true,
    _get_proportionate_gamma_true,
)


def inference_on_test_set(
    models: Dict[str, object],
    scaler: StandardScaler,
    test_data: Sequence[Tuple[np.ndarray, np.ndarray, dict]],
    threshold: float,
    ref_efficiencies_dict: Optional[Dict[str, Sequence[float]]] = None,
) -> Tuple[
    Dict[str, List[float]],
    List[float],
    List[int],
    Dict[str, List[Dict[str, float]]],
]:
    """Run models on test pseudo-experiments and collect predictions.

    Returns
    -------
    mu_hat_test : dict
        model_name → list of mu-hat values.
    mu_true_list : list of float
    gamma_true_list : list of int
    performance_metrics : dict
        model_name → list of per-experiment classification metric dicts.
    """
    mu_hat_test: Dict[str, List[float]] = {n: [] for n in models}
    performance_metrics: Dict[str, List[Dict[str, float]]] = {n: [] for n in models}
    mu_true_list: List[float] = []
    gamma_true_list: List[int] = []

    for X_test, y_test, meta_dict in tqdm(test_data, desc="Inference on test set"):
        X_scaled = scaler.transform(X_test)

        gamma_true = _get_proportionate_gamma_true(meta_dict)
        beta_true = _get_proportionate_beta_true(meta_dict)
        mu_true = meta_dict["mu_true"]
        mu_true_list.append(float(mu_true))
        gamma_true_list.append(int(gamma_true))

        if gamma_true == 0:
            continue

        if ref_efficiencies_dict is None:
            ref_efficiencies_dict = {name: (1.0, 1.0) for name in models}

        for name, model in models.items():
            y_proba = model.predict_proba(X_scaled)[:, 1]
            y_pred = (y_proba > threshold).astype(int)

            _metrics = {
                "accuracy": float(model.score(X_scaled, y_test)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            }
            performance_metrics[name].append(_metrics)

            n_pred = int(np.sum(y_pred))
            expected_signal = _get_expected_signal(
                gamma_true, ref_efficiencies_dict[name][0]
            )
            expected_background = _get_expected_background(
                beta_true, ref_efficiencies_dict[name][1]
            )
            mu_hat = (
                (n_pred - expected_background) / expected_signal
                if expected_signal > 0
                else 0.0
            )
            mu_hat_test[name].append(mu_hat)

    return mu_hat_test, mu_true_list, gamma_true_list, performance_metrics


def evaluate_on_test_set(
    models: Dict[str, object],
    scaler: StandardScaler,
    test_data: Sequence[Tuple[np.ndarray, np.ndarray, dict]],
    threshold: float,
    calib_config: CalibrationConfig,
    eval_config: EvaluationConfig,
    calibration_result: Optional[CalibrationResult] = None,
    output_dir: Optional[Path] = None,
    ref_efficiencies_dict: Optional[Dict[str, Sequence[float]]] = None,
    ctx: Optional["RunContext"] = None,
) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """Full evaluation: inference → CIs → performance + calibration metrics.

    Parameters
    ----------
    calibration_result : CalibrationResult, optional
        If provided, confidence intervals and calibration quality metrics
        are computed.  If ``None``, only classification metrics are
        produced.
    output_dir : Path, optional
        When given, all artifacts are persisted under ``stats/`` and
        ``plots/`` subdirectories.

    Returns
    -------
    results : dict
        Nested dict: ``{model_name: {"performance": {...}, "calibration": {...}}}``.
    raw_data : dict
        ``{model_name: {"mu_hat": List[float], "lower": ndarray|None,
        "upper": ndarray|None, "mu_true": List[float]}}``.
        Used by the plot pipeline (Phase 4).
    """
    # 1. Inference
    mu_hat_test, mu_true_list, gamma_true_list, per_exp_metrics = inference_on_test_set(
        models,
        scaler,
        test_data,
        threshold,
        ref_efficiencies_dict=ref_efficiencies_dict,
    )

    results: Dict[str, dict] = {}

    # raw per-model arrays for Phase-4 plot generation
    raw_data: Dict[str, dict] = {
        name: {
            "mu_hat": mu_hat_test.get(name, []),
            "lower": None,
            "upper": None,
            "mu_true": mu_true_list,
        }
        for name in models
    }

    for model_name in models:
        entry: dict = {"performance": {}, "calibration": {}}

        # --- aggregate per-experiment classification metrics ---
        exp_metrics_list = per_exp_metrics.get(model_name, [])
        if exp_metrics_list:
            agg = {}
            for key in exp_metrics_list[0]:
                agg[key] = float(np.mean([m[key] for m in exp_metrics_list]))
            entry["performance"] = agg

        # --- confidence intervals & calibration quality ---
        if calibration_result is not None and model_name in calibration_result.scores:
            mu_hat_values = mu_hat_test[model_name]
            scores_arr = calibration_result.scores[model_name]

            if calib_config.target == "mu_hat":
                lower, upper = compute_confidence_interval(
                    np.array(mu_hat_values),
                    scores_arr,
                    calib_config.alpha,
                    how=calib_config.how,
                    ci_type=calib_config.ci_type,
                )
            elif calib_config.target == "n_pred":
                n_preds = np.array(mu_hat_values) * np.array(gamma_true_list)
                n_lower, n_upper = compute_confidence_interval(
                    n_preds,
                    scores_arr,
                    calib_config.alpha,
                    how=calib_config.how,
                    ci_type=calib_config.ci_type,
                )
                gamma_arr = np.array(gamma_true_list, dtype=np.float64)
                gamma_arr = np.where(gamma_arr == 0, 1.0, gamma_arr)
                lower = n_lower / gamma_arr
                upper = n_upper / gamma_arr
            else:
                raise ValueError(f"Unknown target: {calib_config.target!r}")

            calib_metrics = compute_calibration_metrics(
                lower,
                upper,
                np.array(mu_true_list),
                calib_config.alpha,
                lambda_=eval_config.ci_score_lambda,
                epsilon=eval_config.ci_score_epsilon,
            )
            entry["calibration"] = calib_metrics
            raw_data[model_name]["lower"] = lower
            raw_data[model_name]["upper"] = upper

            # --- artifact persistence ---
            if output_dir and eval_config.save_artifacts:
                stats_dir = Path(output_dir) / "stats"
                plots_dir = Path(output_dir) / "plots"
                stats_dir.mkdir(parents=True, exist_ok=True)
                plots_dir.mkdir(parents=True, exist_ok=True)

                # CI plot
                plot_confidence_intervals(
                    mu_hat_values,
                    lower,
                    upper,
                    mu_true_list,
                    model_name,
                    calib_metrics["coverage"],
                    output_dir=stats_dir,
                )
                if ctx is not None:
                    import datetime as _dt

                    plot_id = _dt.date.today().strftime("%Y%m%d")
                    ctx.save_artifact(
                        f"stats/test_CI_plots-{plot_id}.png",
                        type="plot",
                        format="png",
                        description="Confidence interval plot (test set)",
                    )

                # CI table
                ci_df = pd.DataFrame(
                    {
                        "model": model_name,
                        "experiment": list(range(len(mu_hat_values))),
                        "y_pred": mu_hat_values,
                        "lower": lower,
                        "upper": upper,
                        "y_true": mu_true_list[: len(mu_hat_values)],
                    }
                )
                ci_df.to_csv(
                    stats_dir / "confidence_intervals.csv",
                    index=False,
                )
                if ctx is not None:
                    ctx.save_artifact(
                        "stats/confidence_intervals.csv",
                        type="scores",
                        format="csv",
                        description="Confidence intervals per experiment (test set)",
                    )

        results[model_name] = entry

    # --- save aggregate metrics ---
    if output_dir and eval_config.save_artifacts:
        stats_dir = Path(output_dir) / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)

        perf_out = {
            name: results[name]["performance"]
            for name in results
            if results[name]["performance"]
        }
        with open(stats_dir / "performance_metrics.json", "w") as fh:
            json.dump(perf_out, fh, indent=2)
        if ctx is not None:
            ctx.save_artifact(
                "stats/performance_metrics.json",
                type="metric",
                format="json",
                description="Aggregate classification metrics on test set",
            )

        calib_out = {
            name: results[name]["calibration"]
            for name in results
            if results[name]["calibration"]
        }
        if calib_out:
            with open(stats_dir / "calibration_metrics.json", "w") as fh:
                json.dump(calib_out, fh, indent=2)
            if ctx is not None:
                ctx.save_artifact(
                    "stats/calibration_metrics.json",
                    type="metric",
                    format="json",
                    description="Calibration quality metrics (coverage, width, ci_score)",
                )

    return results, raw_data
