"""Reusable training orchestrator for the conformal-prediction pipeline.

The ``Trainer`` class exposes three composable stages:

1. :meth:`train` — data loading, scaling, fitting.
2. :meth:`calibrate` — nonconformity scores, quantiles, and CI construction.
3. :meth:`evaluate` — test inference, classification metrics, calibration
   quality metrics.

Each stage can be invoked independently.  :meth:`run` chains all three
for convenience and backward compatibility.

Usage::

    from conformal_predictions.config import load_training_config
    from conformal_predictions.mlops.run_context import RunContext
    from conformal_predictions.training.trainer import Trainer

    cfg = load_training_config("configs/train_toy.yaml")
    ctx = RunContext.create(cfg, config_path="configs/train_toy.yaml")
    trainer = Trainer(cfg, ctx)
    trainer.run()

    # Or individually:
    trainer.train()
    cal_result = trainer.calibrate()
    eval_results = trainer.evaluate(calibration_result=cal_result)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from conformal_predictions.calibration.strategies import (
    CalibrationResult,
    run_calibration,
)
from conformal_predictions.config import (
    CalibrationConfig,
    EvaluationConfig,
    TrainingConfig,
)
from conformal_predictions.data.toy import load_pseudo_experiment
from conformal_predictions.data_viz import contourplot_data
from conformal_predictions.evaluation.error_analysis import (
    build_top_errors_table,
    build_top_errors_wandb_table,
)
from conformal_predictions.evaluation.metrics import compute_performance_metrics
from conformal_predictions.evaluation.plots import (
    plot_ci_coverage,
    plot_ci_width_distribution,
    plot_confusion_matrix,
    plot_distribution,
    plot_mu_hat_distribution,
    plot_nonconformity_by_class,
    plot_nonconformity_ecdf,
    plot_nonconformity_scores,
    plot_pr_curve,
    plot_predictions_ecdf,
    plot_roc_curve,
    plot_target_distribution,
)
from conformal_predictions.evaluation.pseudoexperiments import evaluate_on_test_set
from conformal_predictions.evaluation.reports import generate_run_report
from conformal_predictions.mlops.log_keys import (
    CALIBRATION,
    EDA,
    ERROR_ANALYSIS,
    EVALUATION,
    wandb_key,
)
from conformal_predictions.mlops.run_context import RunContext
from conformal_predictions.training.core import (
    compute_model_efficiencies,
    evaluate_models,
    get_events_count,
    list_split_files,
)
from conformal_predictions.training.models import build_model

# Optional tracker import (Phase 3)
try:
    from conformal_predictions.mlops.tracker import Tracker as _Tracker
except ImportError:
    _Tracker = None  # type: ignore[assignment,misc]


class Trainer:
    """Config-driven training orchestrator.

    Parameters
    ----------
    config : TrainingConfig
        Frozen training configuration.
    run_ctx : RunContext
        Run metadata & output directories.
    tracker : Tracker, optional
        Metric tracker (Phase 3).  When provided, scalar metrics from
        all stages are forwarded to it.
    """

    def __init__(
        self,
        config: TrainingConfig,
        run_ctx: RunContext,
        tracker: Optional[object] = None,
    ) -> None:
        self.config = config
        self.run_ctx = run_ctx
        self.tracker = tracker
        self.models: Dict[str, object] = {}
        self.scaler: Optional[StandardScaler] = None

        # Populated by load_data / train
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._X_val: Optional[np.ndarray] = None
        self._y_val: Optional[np.ndarray] = None
        self._X_train_unscaled: Optional[np.ndarray] = None
        self._calib_data: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
        self._calib_meta: Optional[List[dict]] = None
        self._test_files: Optional[list] = None
        self._ref_efficiencies: Optional[Dict[str, Tuple[float, float]]] = None

        # Populated by train() — train-set predictions per model
        self._train_predictions: Dict[str, np.ndarray] = {}
        self._train_pred_labels: Dict[str, np.ndarray] = {}

        # Populated by evaluate() — used by _generate_plots_and_report
        self._raw_eval_data: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(
        self,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[Tuple[np.ndarray, np.ndarray]],
        List[dict],
        list,
    ]:
        """Load and split data for the toy pipeline.

        Returns
        -------
        X_train, y_train, X_val, y_val, calib_data, calib_meta, test_files
        """
        cfg = self.config
        train_files, val_files, calib_files, test_files = list_split_files(
            Path(cfg.data_dir),
            cfg.mu,
            cfg.test_prefixes,
            cfg.n_test_experiments,
            cfg.valid_size,
            cfg.calib_size,
            cfg.seed,
        )

        train_blocks: List[np.ndarray] = []
        train_labels: List[np.ndarray] = []
        val_blocks: List[np.ndarray] = []
        val_labels: List[np.ndarray] = []
        calib_data: List[Tuple[np.ndarray, np.ndarray]] = []
        calib_meta: List[dict] = []

        for fp in train_files:
            X, y, _ = load_pseudo_experiment(fp)
            train_blocks.append(X)
            train_labels.append(y)

        for fp in val_files:
            X, y, _ = load_pseudo_experiment(fp)
            val_blocks.append(X)
            val_labels.append(y)

        for fp in calib_files:
            X, y, meta = load_pseudo_experiment(fp)
            calib_data.append((X, y))
            calib_meta.append(meta)

        X_train = np.vstack(train_blocks)
        y_train = np.concatenate(train_labels)
        X_val = np.vstack(val_blocks)
        y_val = np.concatenate(val_labels)

        return X_train, y_train, X_val, y_val, calib_data, calib_meta, test_files

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Build and train the configured model."""
        self.models = build_model(self.config.model, self.config.seed)
        for model in tqdm(self.models.values(), desc="Training models"):
            model.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # Stage 1: train
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Data loading, scaling, fitting, and validation evaluation.

        After calling this method, ``self.models``, ``self.scaler``, and
        the internal data splits are populated and available for
        subsequent stages.
        """
        cfg = self.config
        ctx = self.run_ctx
        ctx.ensure_dirs()
        np.random.seed(cfg.seed)

        X_train, y_train, X_val, y_val, calib_data, calib_meta, test_files = (
            self.load_data()
        )

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Store for later stages
        self._X_train = X_train_scaled
        self._y_train = y_train
        self._X_val = X_val_scaled
        self._y_val = y_val
        self._X_train_unscaled = X_train
        self._calib_data = calib_data
        self._calib_meta = calib_meta
        self._test_files = test_files

        print("Starting training on:")
        print(
            f"  X_train shape: {X_train_scaled.shape}, y_train shape: {y_train.shape}"
        )
        print(f"  X_val shape: {X_val_scaled.shape}, y_val shape: {y_val.shape}")
        _pos = int(np.sum(y_train))
        _neg = len(y_train) - _pos
        print(
            f"  y_train: {_pos} positives ({100*_pos/len(y_train):.1f}%), "
            f"{_neg} negatives ({100*_neg/len(y_train):.1f}%)"
        )
        _pos_v = int(np.sum(y_val))
        _neg_v = len(y_val) - _pos_v
        print(
            f"  y_val: {_pos_v} positives ({100*_pos_v/len(y_val):.1f}%), "
            f"{_neg_v} negatives ({100*_neg_v/len(y_val):.1f}%)"
        )

        contourplot_data(X_val, y_val, output_dir=ctx.plots_dir)
        # Register contour plot as EDA artifact
        contour_path = ctx.plots_dir / "data_contour_classes.png"
        if contour_path.exists():
            ctx.save_artifact(
                "plots/data_contour_classes.png",
                type="plot",
                format="png",
                description="Contour plot of data features",
            )
            if self.tracker is not None:
                self.tracker.log_image(
                    wandb_key(EDA, "plots", "contour_plot"), contour_path
                )

        # ---- fit ----
        self.fit(X_train_scaled, y_train)

        # ---- compute and store train predictions ----
        for name, model in self.models.items():
            y_proba_train = model.predict_proba(X_train_scaled)[:, 1]
            self._train_predictions[name] = y_proba_train
            self._train_pred_labels[name] = (y_proba_train > cfg.threshold).astype(int)

        # ---- EDA logging ----
        self._log_eda(X_train, y_train, X_train_scaled)

        # ---- quick validation metrics ----
        perf = evaluate_models(self.models, X_val_scaled, y_val)
        for name, m in perf.items():
            print(f"\n\n{name} validation metrics:")
            print(
                f"\tAccuracy: {m['accuracy']:.4f}"
                f"\tPrecision: {m['precision']:.4f}"
                f"\tRecall: {m['recall']:.4f}"
                f"\tF1: {m['f1']:.4f}"
            )
            if self.tracker is not None:
                for metric_name, value in m.items():
                    if isinstance(value, (int, float)):
                        self.tracker.log(
                            wandb_key(EVALUATION, "val", metric_name),
                            float(value),
                            stage="train",
                        )

        # ---- log train-set performance metrics ----
        for name in self.models:
            y_proba_train = self._train_predictions[name]
            y_pred_train = self._train_pred_labels[name]
            train_perf = compute_performance_metrics(
                y_train, y_pred_train, y_proba_train
            )
            if self.tracker is not None:
                for metric_name, value in train_perf.items():
                    if isinstance(value, (int, float)):
                        self.tracker.log(
                            wandb_key(EVALUATION, "train", metric_name),
                            float(value),
                            stage="train",
                        )

        counts = get_events_count(self.models, X_val_scaled, cfg.threshold)
        for name, count in counts.items():
            print(
                f"{name} N signal events (p_pred > {cfg.threshold}): "
                f"{count} / {int(np.sum(y_val))} (true)"
            )

        # Compute reference efficiencies from the validation set
        self._ref_efficiencies = compute_model_efficiencies(
            self.models, X_val_scaled, y_val, cfg.threshold
        )
        for name, (eps_s, eps_b) in self._ref_efficiencies.items():
            print(
                f"{name} reference efficiencies: eps_signal={eps_s:.4f}, eps_background={eps_b:.4f}"
            )

    # ------------------------------------------------------------------
    # Stage 2: calibrate
    # ------------------------------------------------------------------

    def calibrate(
        self,
        calib_config: Optional[CalibrationConfig] = None,
    ) -> CalibrationResult:
        """Run conformal calibration on the calibration set.

        Parameters
        ----------
        calib_config : CalibrationConfig, optional
            Overrides the config embedded in ``self.config.calibration``.

        Returns
        -------
        CalibrationResult
        """
        if calib_config is None:
            calib_config = self.config.calibration

        if self.models is None or self.scaler is None:
            raise RuntimeError(
                "Models and scaler must be fitted before calibration. "
                "Call trainer.train() first."
            )
        if self._calib_data is None:
            raise RuntimeError(
                "Calibration data not loaded. Call trainer.train() first."
            )

        print("\nRunning calibration...")
        print(f"  {len(self._calib_data)} calibration experiments")
        print(f"  target={calib_config.target}  how={calib_config.how}")
        print(f"  alpha={calib_config.alpha:.4f}  ci_type={calib_config.ci_type}")

        # Use reference efficiencies from the (single) model.
        ref_efficiencies: Optional[Tuple[float, float]] = None
        if self._ref_efficiencies:
            # Single-model pipeline: take the only entry directly.
            eps_s, eps_b = next(iter(self._ref_efficiencies.values()))
            ref_efficiencies = (float(eps_s), float(eps_b))

        result = run_calibration(
            self.models,
            self.scaler,
            self._calib_data,
            self._calib_meta,
            self.config.threshold,
            calib_config,
            output_dir=self.run_ctx.output_dir,
            ref_efficiencies=ref_efficiencies,
            ctx=self.run_ctx,
        )

        for name, scores_arr in result.scores.items():
            mu = float(np.mean(scores_arr)) if len(scores_arr) else float("nan")
            sd = float(np.std(scores_arr)) if len(scores_arr) else float("nan")
            print(f"  {name} score stats: mean={mu:.4f} ± std={sd:.4f}")
            if self.tracker is not None:
                self.tracker.log(
                    wandb_key(CALIBRATION, "nonconformity", "score_mean"),
                    mu,
                    stage="calibrate",
                )
                self.tracker.log(
                    wandb_key(CALIBRATION, "nonconformity", "score_std"),
                    sd,
                    stage="calibrate",
                )

        if result.quantiles:
            for name, (ql, qh) in result.quantiles.items():
                print(f"  {name} quantiles: q_low={ql:.4f}  q_high={qh:.4f}")
                width = qh - ql
                if self.tracker is not None:
                    self.tracker.log(
                        wandb_key(CALIBRATION, "nonconformity", "q_low"),
                        ql,
                        stage="calibrate",
                    )
                    self.tracker.log(
                        wandb_key(CALIBRATION, "nonconformity", "q_high"),
                        qh,
                        stage="calibrate",
                    )
                    self.tracker.log(
                        wandb_key(CALIBRATION, "nonconformity", "width"),
                        width,
                        stage="calibrate",
                    )

        # Log calibration config
        if self.tracker is not None:
            self.tracker.log(
                wandb_key(CALIBRATION, "config", "alpha"),
                calib_config.alpha,
                stage="calibrate",
            )

        return result

    # ------------------------------------------------------------------
    # Stage 3: evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        calibration_result: Optional[CalibrationResult] = None,
        eval_config: Optional[EvaluationConfig] = None,
    ) -> Dict[str, dict]:
        """Evaluate on the test set.

        Parameters
        ----------
        calibration_result : CalibrationResult, optional
            When provided, confidence intervals and calibration quality
            metrics are computed.
        eval_config : EvaluationConfig, optional
            Overrides ``self.config.evaluation``.

        Returns
        -------
        dict
            ``{model_name: {"performance": {...}, "calibration": {...}}}``
        """
        if eval_config is None:
            eval_config = self.config.evaluation

        if self.models is None or self.scaler is None:
            raise RuntimeError(
                "Models and scaler must be fitted before evaluation. "
                "Call trainer.train() first."
            )
        if self._test_files is None:
            raise RuntimeError("Test files not loaded. Call trainer.train() first.")

        print("\nRunning evaluation on test set...")

        test_data = []
        for fp in self._test_files:
            X_t, y_t, meta_t = load_pseudo_experiment(fp)
            test_data.append((X_t, y_t, meta_t))

        results, raw_data = evaluate_on_test_set(
            self.models,
            self.scaler,
            test_data,
            self.config.threshold,
            calib_config=self.config.calibration,
            eval_config=eval_config,
            calibration_result=calibration_result,
            output_dir=self.run_ctx.output_dir,
            ctx=self.run_ctx,
        )

        # Store raw arrays; add val-set probabilities for ROC/PR plots
        self._raw_eval_data = raw_data
        if self._X_val is not None and self._y_val is not None:
            for name, model in self.models.items():
                y_proba = model.predict_proba(self._X_val)[:, 1]
                self._raw_eval_data.setdefault(name, {})["y_true_val"] = self._y_val
                self._raw_eval_data[name]["y_proba_val"] = y_proba

        for name, res in results.items():
            perf = res.get("performance", {})
            cal = res.get("calibration", {})
            print(f"\n  {name}:")
            if perf:
                parts = [
                    f"{k}={v:.4f}" for k, v in perf.items() if isinstance(v, float)
                ]
                print(f"    performance: {', '.join(parts)}")
            if cal:
                print(
                    f"    coverage={cal.get('coverage', 0):.4f}"
                    f"  width={cal.get('width', 0):.4f}"
                    f"  ci_score={cal.get('ci_score', 0):.4f}"
                )
            if self.tracker is not None:
                for metric_name, value in perf.items():
                    if isinstance(value, (int, float)):
                        self.tracker.log(
                            wandb_key(EVALUATION, "test", metric_name),
                            float(value),
                            stage="evaluate",
                        )
                for metric_name, value in cal.items():
                    if isinstance(value, (int, float)):
                        self.tracker.log(
                            wandb_key(CALIBRATION, "metrics", metric_name),
                            float(value),
                            stage="evaluate",
                        )

        # Generate plots and report if enabled
        if getattr(self.config, "reporting", None) is not None:
            if self.config.reporting.generate_plots:
                self._generate_plots_and_report(results, calibration_result)

        return results

    # ------------------------------------------------------------------
    # Plots + report (Phase 4.5)
    # ------------------------------------------------------------------

    def _generate_plots_and_report(
        self,
        eval_results: Dict[str, dict],
        calibration_result: Optional[CalibrationResult],
    ) -> None:
        """Generate per-run plots, register artifacts, and write report.md."""
        ctx = self.run_ctx
        dpi = self.config.reporting.figure_dpi
        alpha = self.config.calibration.alpha

        coverages: Dict[str, float] = {}

        for model_name in self.models:
            safe_name = model_name.replace(" ", "_")
            raw = self._raw_eval_data.get(model_name, {})

            # Joint train+val ROC curve
            y_true_train = self._y_train if self._y_train is not None else None
            y_proba_train = self._train_predictions.get(model_name)

            if "y_true_val" in raw and "y_proba_val" in raw:
                path = ctx.plots_dir / "roc_curve.png"
                plot_roc_curve(
                    raw["y_true_val"],
                    raw["y_proba_val"],
                    model_name,
                    output_path=path,
                    dpi=dpi,
                    y_true_train=y_true_train,
                    y_score_train=y_proba_train,
                )
                ctx.save_artifact(
                    "plots/roc_curve.png",
                    type="plot",
                    format="png",
                    description=f"ROC curve — {model_name}",
                )
                if self.tracker is not None:
                    self.tracker.log_image(
                        wandb_key(EVALUATION, "plots", "roc_curve"), path
                    )

            # Joint train+val PR curve
            if "y_true_val" in raw and "y_proba_val" in raw:
                path = ctx.plots_dir / "pr_curve.png"
                plot_pr_curve(
                    raw["y_true_val"],
                    raw["y_proba_val"],
                    model_name,
                    output_path=path,
                    dpi=dpi,
                    y_true_train=y_true_train,
                    y_score_train=y_proba_train,
                )
                ctx.save_artifact(
                    "plots/pr_curve.png",
                    type="plot",
                    format="png",
                    description=f"PR curve — {model_name}",
                )
                if self.tracker is not None:
                    self.tracker.log_image(
                        wandb_key(EVALUATION, "plots", "pr_curve"), path
                    )

            # Predictions ECDF (train vs val)
            if "y_proba_val" in raw and y_proba_train is not None:
                path = ctx.plots_dir / "predictions_ecdf.png"
                plot_predictions_ecdf(
                    y_proba_train,
                    raw["y_proba_val"],
                    output_path=path,
                    model_name=model_name,
                    dpi=dpi,
                )
                ctx.save_artifact(
                    "plots/predictions_ecdf.png",
                    type="plot",
                    format="png",
                    description=f"Predictions ECDF — {model_name}",
                )
                if self.tracker is not None:
                    self.tracker.log_image(
                        wandb_key(EVALUATION, "plots", "predictions_ecdf"),
                        path,
                    )

            # μ̂ distribution (test set)
            mu_hat_vals = raw.get("mu_hat", [])
            if mu_hat_vals:
                path = ctx.plots_dir / "mu_hat_distribution_test.png"
                plot_mu_hat_distribution(
                    mu_hat_vals,
                    output_path=path,
                    model_name=model_name,
                    dpi=dpi,
                )
                ctx.save_artifact(
                    "plots/mu_hat_distribution_test.png",
                    type="plot",
                    format="png",
                    description=f"μ̂ distribution (test set) — {model_name}",
                )
                if self.tracker is not None:
                    self.tracker.log_image(
                        wandb_key(CALIBRATION, "plots", "mu_hat_distribution_test"),
                        path,
                    )

            # CI width distribution
            lower = raw.get("lower")
            upper = raw.get("upper")
            if lower is not None and upper is not None:
                widths = np.asarray(upper) - np.asarray(lower)
                path = ctx.plots_dir / "ci_width_distribution.png"
                plot_ci_width_distribution(
                    widths,
                    output_path=path,
                    model_name=model_name,
                    dpi=dpi,
                )
                ctx.save_artifact(
                    "plots/ci_width_distribution.png",
                    type="plot",
                    format="png",
                    description=f"CI width distribution — {model_name}",
                )
                if self.tracker is not None:
                    self.tracker.log_image(
                        wandb_key(CALIBRATION, "plots", "ci_width_distribution"),
                        path,
                    )

            # Collect coverage for CI chart
            cal = eval_results.get(model_name, {}).get("calibration", {})
            if "coverage" in cal:
                coverages[model_name] = cal["coverage"]

        # CI coverage chart
        if coverages:
            target_cov = 1.0 - alpha
            path = ctx.plots_dir / "ci_coverage.png"
            plot_ci_coverage(coverages, target_cov, output_path=path, dpi=dpi)
            ctx.save_artifact(
                "plots/ci_coverage.png",
                type="plot",
                format="png",
                description="CI coverage",
            )
            if self.tracker is not None:
                self.tracker.log_image(
                    wandb_key(CALIBRATION, "plots", "ci_coverage"), path
                )

        # Calibration-specific plots
        if calibration_result is not None:
            self._log_calibration_plots(calibration_result)
            # Log calibration-set performance metrics
            self._log_calibration_metrics(calibration_result)

        # Error analysis
        self._log_error_analysis()

        # Markdown report
        report_path = ctx.output_dir / "report.md"
        generate_run_report(ctx, metrics=eval_results, output_path=report_path)
        ctx.save_artifact(
            "report.md",
            type="report",
            format="md",
            description="Run summary report",
        )
        print(f"\nReport saved to {report_path}")

    # ------------------------------------------------------------------
    # EDA logging (Phase 4.5)
    # ------------------------------------------------------------------

    def _log_eda(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_train_scaled: np.ndarray,
    ) -> None:
        """Log EDA metrics and plots."""
        ctx = self.run_ctx
        dpi = getattr(getattr(self.config, "reporting", None), "figure_dpi", 150)

        # Class balance scalar
        class_balance = float(np.mean(y_train))
        if self.tracker is not None:
            self.tracker.log(
                wandb_key(EDA, "train", "class_balance"),
                class_balance,
                stage="train",
            )

        # Target distribution plot
        path = ctx.plots_dir / "target_distribution.png"
        plot_target_distribution(y_train, output_path=path, dpi=dpi)
        ctx.save_artifact(
            "plots/target_distribution.png",
            type="plot",
            format="png",
            description="Target class distribution (train)",
        )
        if self.tracker is not None:
            self.tracker.log_image(wandb_key(EDA, "plots", "target_distribution"), path)

    # ------------------------------------------------------------------
    # Calibration plots (Phase 4.5)
    # ------------------------------------------------------------------

    def _log_calibration_plots(self, calibration_result: CalibrationResult) -> None:
        """Generate and log calibration-specific plots."""
        ctx = self.run_ctx
        dpi = self.config.reporting.figure_dpi
        alpha = self.config.calibration.alpha

        for model_name in self.models:
            safe_name = model_name.replace(" ", "_")
            scores = calibration_result.scores.get(model_name)

            if scores is not None and len(scores) > 0:
                # Nonconformity distribution (histogram)
                path = ctx.plots_dir / "nonconformity_scores.png"
                plot_nonconformity_scores(
                    scores,
                    alpha,
                    output_path=path,
                    model_name=model_name,
                    dpi=dpi,
                )
                ctx.save_artifact(
                    "plots/nonconformity_scores.png",
                    type="plot",
                    format="png",
                    description=f"Nonconformity scores — {model_name}",
                )
                if self.tracker is not None:
                    self.tracker.log_image(
                        wandb_key(CALIBRATION, "plots", "nonconformity_distribution"),
                        path,
                    )

                # Nonconformity ECDF
                path_ecdf = ctx.plots_dir / "nonconformity_ecdf.png"
                plot_nonconformity_ecdf(
                    scores,
                    output_path=path_ecdf,
                    model_name=model_name,
                    dpi=dpi,
                )
                ctx.save_artifact(
                    "plots/nonconformity_ecdf.png",
                    type="plot",
                    format="png",
                    description=f"Nonconformity ECDF — {model_name}",
                )
                if self.tracker is not None:
                    self.tracker.log_image(
                        wandb_key(CALIBRATION, "plots", "nonconformity_ecdf"),
                        path_ecdf,
                    )

            # Nonconformity by class (if calib labels available)
            if (
                calibration_result.calib_y_true is not None
                and calibration_result.calib_y_proba is not None
                and model_name in calibration_result.calib_y_proba
            ):
                y_calib = calibration_result.calib_y_true
                proba_calib = calibration_result.calib_y_proba[model_name]
                # Use prediction scores as proxy for per-sample nonconformity
                mask0 = y_calib == 0
                mask1 = y_calib == 1
                path_bc = ctx.plots_dir / "nonconformity_by_class.png"
                plot_nonconformity_by_class(
                    proba_calib[mask0],
                    proba_calib[mask1],
                    output_path=path_bc,
                    model_name=model_name,
                    dpi=dpi,
                )
                ctx.save_artifact(
                    "plots/nonconformity_by_class.png",
                    type="plot",
                    format="png",
                    description=f"Nonconformity by class — {model_name}",
                )
                if self.tracker is not None:
                    self.tracker.log_image(
                        wandb_key(
                            CALIBRATION,
                            "plots",
                            "nonconformity_distribution_by_class",
                        ),
                        path_bc,
                    )

            # Per-block q_low / q_high / width distributions
            q_lows = calibration_result.per_block_q_low.get(model_name, [])
            q_highs = calibration_result.per_block_q_high.get(model_name, [])
            if q_lows and q_highs:
                # q_low distribution
                path_ql = ctx.plots_dir / "q_low_distribution.png"
                plot_distribution(
                    q_lows,
                    output_path=path_ql,
                    title=f"q_low Distribution — {model_name}",
                    xlabel="q_low",
                    color="#4E79A7",
                    dpi=dpi,
                )
                ctx.save_artifact(
                    "plots/q_low_distribution.png",
                    type="plot",
                    format="png",
                    description=f"q_low distribution — {model_name}",
                )
                if self.tracker is not None:
                    self.tracker.log_image(
                        wandb_key(CALIBRATION, "plots", "q_low_distribution"),
                        path_ql,
                    )

                # q_high distribution
                path_qh = ctx.plots_dir / "q_high_distribution.png"
                plot_distribution(
                    q_highs,
                    output_path=path_qh,
                    title=f"q_high Distribution — {model_name}",
                    xlabel="q_high",
                    color="#E15759",
                    dpi=dpi,
                )
                ctx.save_artifact(
                    "plots/q_high_distribution.png",
                    type="plot",
                    format="png",
                    description=f"q_high distribution — {model_name}",
                )
                if self.tracker is not None:
                    self.tracker.log_image(
                        wandb_key(CALIBRATION, "plots", "q_high_distribution"),
                        path_qh,
                    )

                # CI width distribution (per-block)
                block_widths = [qh - ql for ql, qh in zip(q_lows, q_highs)]
                path_bw = ctx.plots_dir / "block_ci_width_distribution.png"
                plot_distribution(
                    block_widths,
                    output_path=path_bw,
                    title=f"Per-block CI Width — {model_name}",
                    xlabel="CI Width",
                    color="mediumpurple",
                    dpi=dpi,
                )
                ctx.save_artifact(
                    "plots/block_ci_width_distribution.png",
                    type="plot",
                    format="png",
                    description=f"Per-block CI width distribution — {model_name}",
                )
                if self.tracker is not None:
                    self.tracker.log_image(
                        wandb_key(CALIBRATION, "plots", "block_ci_width_distribution"),
                        path_bw,
                    )

    # ------------------------------------------------------------------
    # Calibration-set metrics (Phase 4.5)
    # ------------------------------------------------------------------

    def _log_calibration_metrics(self, calibration_result: CalibrationResult) -> None:
        """Compute and log performance metrics on the calibration set."""
        if (
            calibration_result.calib_y_true is None
            or calibration_result.calib_y_pred is None
            or calibration_result.calib_y_proba is None
        ):
            return

        for name in self.models:
            if name not in calibration_result.calib_y_pred:
                continue
            y_true = calibration_result.calib_y_true
            y_pred = calibration_result.calib_y_pred[name]
            y_proba = calibration_result.calib_y_proba[name]
            perf = compute_performance_metrics(y_true, y_pred, y_proba)
            if self.tracker is not None:
                for metric_name, value in perf.items():
                    if isinstance(value, (int, float)):
                        self.tracker.log(
                            wandb_key(CALIBRATION, "metrics", metric_name),
                            float(value),
                            stage="calibrate",
                        )

    # ------------------------------------------------------------------
    # Error analysis (Phase 4.5)
    # ------------------------------------------------------------------

    def _log_error_analysis(self) -> None:
        """Generate confusion matrices and top-error tables."""
        ctx = self.run_ctx
        dpi = self.config.reporting.figure_dpi

        for model_name in self.models:
            safe_name = model_name.replace(" ", "_")

            # --- Train error analysis ---
            y_proba_train = self._train_predictions.get(model_name)
            y_pred_train = self._train_pred_labels.get(model_name)
            if (
                self._y_train is not None
                and y_pred_train is not None
                and y_proba_train is not None
            ):
                # Confusion matrix — train
                path_cm = ctx.plots_dir / "train_confusion_matrix.png"
                plot_confusion_matrix(
                    self._y_train,
                    y_pred_train,
                    output_path=path_cm,
                    title=f"Confusion Matrix (Train) — {model_name}",
                    dpi=dpi,
                )
                ctx.save_artifact(
                    "plots/train_confusion_matrix.png",
                    type="plot",
                    format="png",
                    description=f"Confusion matrix (train) — {model_name}",
                )
                if self.tracker is not None:
                    self.tracker.log_image(
                        wandb_key(ERROR_ANALYSIS, "train", "confusion_matrix"),
                        path_cm,
                    )

                # Top errors — train
                df_errors = build_top_errors_table(
                    self._y_train,
                    y_pred_train,
                    y_proba_train,
                    X=self._X_train,
                )
                csv_path = ctx.output_dir / "stats" / "train_top_errors.csv"
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                df_errors.to_csv(csv_path, index=False)
                ctx.save_artifact(
                    "stats/train_top_errors.csv",
                    type="error_analysis",
                    format="csv",
                    description=f"Top errors (train) — {model_name}",
                )
                if self.tracker is not None:
                    tbl = build_top_errors_wandb_table(df_errors)
                    if tbl is not None:
                        self.tracker.log_table(
                            wandb_key(ERROR_ANALYSIS, "train", "top_errors"),
                            tbl,
                        )

            # --- Val / test error analysis ---
            raw = self._raw_eval_data.get(model_name, {})
            y_true_val = raw.get("y_true_val", self._y_val)
            y_proba_val = raw.get("y_proba_val")
            if y_true_val is not None and y_proba_val is not None:
                y_pred_val = (y_proba_val > self.config.threshold).astype(int)

                # Confusion matrix — val/test
                path_cm_v = ctx.plots_dir / "test_confusion_matrix.png"
                plot_confusion_matrix(
                    y_true_val,
                    y_pred_val,
                    output_path=path_cm_v,
                    title=f"Confusion Matrix (Test/Val) — {model_name}",
                    dpi=dpi,
                )
                ctx.save_artifact(
                    "plots/test_confusion_matrix.png",
                    type="plot",
                    format="png",
                    description=f"Confusion matrix (test/val) — {model_name}",
                )
                if self.tracker is not None:
                    self.tracker.log_image(
                        wandb_key(ERROR_ANALYSIS, "test", "confusion_matrix"),
                        path_cm_v,
                    )

                # Top errors — val/test
                df_errors_v = build_top_errors_table(
                    y_true_val,
                    y_pred_val,
                    y_proba_val,
                )
                csv_path_v = ctx.output_dir / "stats" / "test_top_errors.csv"
                csv_path_v.parent.mkdir(parents=True, exist_ok=True)
                df_errors_v.to_csv(csv_path_v, index=False)
                ctx.save_artifact(
                    "stats/test_top_errors.csv",
                    type="error_analysis",
                    format="csv",
                    description=f"Top errors (test/val) — {model_name}",
                )
                if self.tracker is not None:
                    tbl_v = build_top_errors_wandb_table(df_errors_v)
                    if tbl_v is not None:
                        self.tracker.log_table(
                            wandb_key(ERROR_ANALYSIS, "test", "top_errors"),
                            tbl_v,
                        )

    # ------------------------------------------------------------------
    # Full pipeline (backward-compatible)
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the full train → calibrate → evaluate pipeline.

        This is the backward-compatible entry point that chains all three
        stages.  Calibration and evaluation are skipped if their
        respective ``enabled`` flags are ``False`` in the config.
        """
        self.train()

        calibration_result = None
        if self.config.calibration.enabled:
            calibration_result = self.calibrate()

        if self.config.evaluation.enabled:
            self.evaluate(calibration_result=calibration_result)

        # Persist run metadata
        meta_path = self.run_ctx.save_metadata()
        print(f"\nRun metadata saved to {meta_path}")
