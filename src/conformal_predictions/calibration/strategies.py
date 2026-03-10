"""High-level calibration orchestrator.

``run_calibration`` wires together score computation, quantile
extraction, confidence-interval construction, and artifact persistence
into a single composable entry point.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from conformal_predictions.mlops.run_context import RunContext

from conformal_predictions.calibration.intervals import (
    extract_quantiles,
    save_quantiles,
)
from conformal_predictions.calibration.scores import (
    compute_mu_hat,
    compute_nonconformity_scores,
    save_scores,
)
from conformal_predictions.config import CalibrationConfig
from conformal_predictions.data_viz import (
    plot_mu_hat_distribution,
    plot_nonconformity_scores,
)


@dataclass
class CalibrationResult:
    """Container for all calibration outputs.

    Attributes
    ----------
    scores : dict
        model_name → 1-D array of nonconformity scores.
    quantiles : dict
        model_name → (q_low, q_high).
    mu_hat : dict
        model_name → list of calibration mu-hat values.
    mu_hat_stats : dict
        model_name → summary statistics dict.
    config : CalibrationConfig
        The configuration used for this calibration.
    per_block_q_low : dict
        model_name → list of per-experiment q_low values.
    per_block_q_high : dict
        model_name → list of per-experiment q_high values.
    per_block_scores : dict
        model_name → list of per-experiment score arrays.
    calib_y_true : np.ndarray or None
        Concatenated calibration-set true labels.
    calib_y_pred : dict or None
        model_name → concatenated hard predictions on calib set.
    calib_y_proba : dict or None
        model_name → concatenated predicted probabilities on calib set.
    """

    scores: Dict[str, np.ndarray] = field(default_factory=dict)
    quantiles: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    mu_hat: Dict[str, List[float]] = field(default_factory=dict)
    mu_hat_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    config: Optional[CalibrationConfig] = None
    per_block_q_low: Dict[str, List[float]] = field(default_factory=dict)
    per_block_q_high: Dict[str, List[float]] = field(default_factory=dict)
    per_block_scores: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    calib_y_true: Optional[np.ndarray] = None
    calib_y_pred: Optional[Dict[str, np.ndarray]] = None
    calib_y_proba: Optional[Dict[str, np.ndarray]] = None


def run_calibration(
    models: Dict[str, object],
    scaler: StandardScaler,
    calib_data: Sequence[Tuple[np.ndarray, np.ndarray]],
    calib_meta: Sequence[dict],
    threshold: float,
    calib_config: CalibrationConfig,
    output_dir: Optional[Path] = None,
    ref_efficiencies: Optional[Sequence[float]] = None,
    ctx: Optional["RunContext"] = None,
) -> CalibrationResult:
    """Execute the full calibration pipeline.

    Parameters
    ----------
    models : dict
        Trained model catalogue.
    scaler : StandardScaler
        Fitted feature scaler.
    calib_data, calib_meta : sequences
        Calibration pseudo-experiments — ``(X, y)`` pairs and metadata.
    threshold : float
        Classification decision threshold.
    calib_config : CalibrationConfig
        Calibration settings (target, how, alpha, ci_type, …).
    output_dir : Path, optional
        If given, calibration artifacts are persisted here.
    ref_efficiencies : sequence of float, optional
        ``(eps_signal, eps_background)``.

    Returns
    -------
    CalibrationResult
    """
    result = CalibrationResult(config=calib_config)

    # 1. Nonconformity scores (flat + per-block)
    raw_scores = compute_nonconformity_scores(
        models,
        scaler,
        calib_data,
        calib_meta,
        threshold,
        target=calib_config.target,
        how=calib_config.how,
        ref_efficiencies=ref_efficiencies,
    )
    result.scores = {n: np.array(v) for n, v in raw_scores.items()}

    # Collect per-block score arrays for per-block quantile computation
    per_block: Dict[str, List[np.ndarray]] = {n: [] for n in models}
    for name, scores_list in raw_scores.items():
        # Each item is one score per experiment — group into single-element arrays
        for s in scores_list:
            per_block[name].append(np.array([s]))
    result.per_block_scores = per_block

    # 2. Quantile extraction (full-set)
    for name, scores_arr in result.scores.items():
        if len(scores_arr) > 0:
            result.quantiles[name] = extract_quantiles(
                scores_arr,
                calib_config.alpha,
                how=calib_config.how,
                ci_type=calib_config.ci_type,
            )

    # 2b. Per-block quantile computation (sliding window over raw scores)
    #     Since each experiment yields a single score, we compute quantiles
    #     on cumulative sub-windows to show convergence / variability.
    for name, scores_arr in result.scores.items():
        if len(scores_arr) < 2:
            continue
        q_lows: List[float] = []
        q_highs: List[float] = []
        # Use leave-one-out: quantiles from all-but-one experiment
        for i in range(len(scores_arr)):
            subset = np.concatenate([scores_arr[:i], scores_arr[i + 1 :]])
            if len(subset) > 0:
                ql, qh = extract_quantiles(
                    subset,
                    calib_config.alpha,
                    how=calib_config.how,
                    ci_type=calib_config.ci_type,
                )
                q_lows.append(ql)
                q_highs.append(qh)
        result.per_block_q_low[name] = q_lows
        result.per_block_q_high[name] = q_highs

    # 2c. Collect calibration-set predictions for downstream metrics
    all_y_true: List[np.ndarray] = []
    calib_y_pred: Dict[str, List[np.ndarray]] = {n: [] for n in models}
    calib_y_proba: Dict[str, List[np.ndarray]] = {n: [] for n in models}
    for X_calib, y_calib in calib_data:
        X_scaled = scaler.transform(X_calib)
        all_y_true.append(y_calib)
        for name, model in models.items():
            proba = model.predict_proba(X_scaled)[:, 1]
            calib_y_proba[name].append(proba)
            calib_y_pred[name].append((proba > threshold).astype(int))
    result.calib_y_true = np.concatenate(all_y_true)
    result.calib_y_pred = {n: np.concatenate(v) for n, v in calib_y_pred.items()}
    result.calib_y_proba = {n: np.concatenate(v) for n, v in calib_y_proba.items()}

    # 3. mu-hat calibration distribution
    mu_hat, mu_hat_stats = compute_mu_hat(
        models,
        scaler,
        calib_data,
        calib_meta,
        threshold,
        ref_efficiencies=ref_efficiencies or (1.0, 1.0),
    )
    result.mu_hat = mu_hat
    result.mu_hat_stats = mu_hat_stats

    # 4. Artifact persistence
    if output_dir and calib_config.save_artifacts:
        stats_dir = Path(output_dir) / "stats"
        plots_dir = Path(output_dir) / "plots"
        stats_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        scores_filename = f"{calib_config.target}_nonconf_scores.npz"
        save_scores(
            raw_scores,
            stats_dir,
            filename=scores_filename,
        )
        if ctx is not None:
            ctx.save_artifact(
                f"stats/{scores_filename}",
                type="calibration",
                format="npz",
                description="Nonconformity scores (calibration set)",
            )
            csv_name = scores_filename.replace(".npz", "_distribution.csv")
            ctx.save_artifact(
                f"stats/{csv_name}",
                type="calibration",
                format="csv",
                description="Nonconformity score histogram (calibration set)",
            )

        np.savez(
            stats_dir / "mu_hat_calib_distribution.npz",
            **{n: np.array(v) for n, v in mu_hat.items()},
        )
        if ctx is not None:
            ctx.save_artifact(
                "stats/mu_hat_calib_distribution.npz",
                type="calibration",
                format="npz",
                description="mu_hat calibration distribution",
            )

        if mu_hat_stats:
            df_stats = pd.DataFrame(
                [{"Model": n, **mu_hat_stats[n]} for n in mu_hat_stats]
            )
            df_stats.to_csv(stats_dir / "mu_hat_calibration_stats.csv", index=False)
            if ctx is not None:
                ctx.save_artifact(
                    "stats/mu_hat_calibration_stats.csv",
                    type="calibration",
                    format="csv",
                    description="mu_hat calibration summary statistics per model",
                )

        save_quantiles(
            result.quantiles,
            calib_config.alpha,
            calib_config.ci_type,
            calib_config.how,
            stats_dir,
        )
        if ctx is not None:
            ctx.save_artifact(
                "stats/calibration_quantiles.csv",
                type="calibration",
                format="csv",
                description="Conformal quantiles per model",
            )

        # Summary JSON
        summary = {
            "alpha": calib_config.alpha,
            "ci_type": calib_config.ci_type,
            "how": calib_config.how,
            "target": calib_config.target,
            "n_calibration_experiments": len(calib_data),
        }
        for name, scores_arr in result.scores.items():
            summary[f"{name}_score_mean"] = float(np.mean(scores_arr))
            summary[f"{name}_score_std"] = float(np.std(scores_arr))
        with open(stats_dir / "calibration_summary.json", "w") as fh:
            json.dump(summary, fh, indent=2)
        if ctx is not None:
            ctx.save_artifact(
                "stats/calibration_summary.json",
                type="calibration",
                format="json",
                description="Calibration run summary (alpha, quantiles, score stats)",
            )

        # Plots
        plot_nonconformity_scores(
            raw_scores,
            scores_label=calib_config.target,
            output_dir=plots_dir,
        )
        if ctx is not None:
            for model_name in models:
                ctx.save_artifact(
                    f"plots/{calib_config.target}_scores_distribution_{model_name}.png",
                    type="plot",
                    format="png",
                    description=f"Nonconformity score distribution — {model_name}",
                )
            ctx.save_artifact(
                f"plots/{calib_config.target}_scores_distribution_comparison.png",
                type="plot",
                format="png",
                description="Nonconformity score distribution comparison across models",
            )

        if mu_hat and mu_hat_stats:
            plot_mu_hat_distribution(mu_hat, mu_hat_stats, output_dir=plots_dir)
            if ctx is not None:
                for model_name in models:
                    ctx.save_artifact(
                        f"plots/mu_hat_distribution_{model_name}.png",
                        type="plot",
                        format="png",
                        description=f"mu_hat calibration distribution — {model_name}",
                    )

    return result
