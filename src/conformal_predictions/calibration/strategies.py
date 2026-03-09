"""High-level calibration orchestrator.

``run_calibration`` wires together score computation, quantile
extraction, confidence-interval construction, and artifact persistence
into a single composable entry point.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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
    """

    scores: Dict[str, np.ndarray] = field(default_factory=dict)
    quantiles: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    mu_hat: Dict[str, List[float]] = field(default_factory=dict)
    mu_hat_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    config: Optional[CalibrationConfig] = None


def run_calibration(
    models: Dict[str, object],
    scaler: StandardScaler,
    calib_data: Sequence[Tuple[np.ndarray, np.ndarray]],
    calib_meta: Sequence[dict],
    threshold: float,
    calib_config: CalibrationConfig,
    output_dir: Optional[Path] = None,
    ref_efficiencies: Optional[Sequence[float]] = None,
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

    # 1. Nonconformity scores
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

    # 2. Quantile extraction
    for name, scores_arr in result.scores.items():
        if len(scores_arr) > 0:
            result.quantiles[name] = extract_quantiles(
                scores_arr,
                calib_config.alpha,
                how=calib_config.how,
                ci_type=calib_config.ci_type,
            )

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

        save_scores(
            raw_scores,
            stats_dir,
            filename=f"{calib_config.target}_nonconf_scores.npz",
        )

        np.savez(
            stats_dir / "mu_hat_calib_distribution.npz",
            **{n: np.array(v) for n, v in mu_hat.items()},
        )

        if mu_hat_stats:
            df_stats = pd.DataFrame(
                [{"Model": n, **mu_hat_stats[n]} for n in mu_hat_stats]
            )
            df_stats.to_csv(stats_dir / "mu_hat_calibration_stats.csv", index=False)

        save_quantiles(
            result.quantiles,
            calib_config.alpha,
            calib_config.ci_type,
            calib_config.how,
            stats_dir,
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

        # Plots
        plot_nonconformity_scores(
            raw_scores,
            scores_label=calib_config.target,
            output_dir=plots_dir,
        )
        if mu_hat and mu_hat_stats:
            plot_mu_hat_distribution(mu_hat, mu_hat_stats, output_dir=plots_dir)

    return result
