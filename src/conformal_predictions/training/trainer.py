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
from conformal_predictions.evaluation.pseudoexperiments import evaluate_on_test_set
from conformal_predictions.mlops.run_context import RunContext
from conformal_predictions.training.core import (
    compute_model_efficiencies,
    evaluate_models,
    get_events_count,
    list_split_files,
)
from conformal_predictions.training.models import build_default_models

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
        self._calib_data: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
        self._calib_meta: Optional[List[dict]] = None
        self._test_files: Optional[list] = None
        self._ref_efficiencies: Optional[Dict[str, Tuple[float, float]]] = None

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
        """Build and train the default model catalogue."""
        self.models = build_default_models(self.config.seed)
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

        # ---- fit ----
        self.fit(X_train_scaled, y_train)

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
                            f"{name}.val_{metric_name}",
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

        # Average ref efficiencies across models (per-model support would
        # require a refactor of run_calibration's API).
        ref_efficiencies: Optional[Tuple[float, float]] = None
        if self._ref_efficiencies:
            avg_eps_s = float(np.mean([v[0] for v in self._ref_efficiencies.values()]))
            avg_eps_b = float(np.mean([v[1] for v in self._ref_efficiencies.values()]))
            ref_efficiencies = (avg_eps_s, avg_eps_b)

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
                self.tracker.log(f"{name}.calib_score_mean", mu, stage="calibrate")
                self.tracker.log(f"{name}.calib_score_std", sd, stage="calibrate")

        if result.quantiles:
            for name, (ql, qh) in result.quantiles.items():
                print(f"  {name} quantiles: q_low={ql:.4f}  q_high={qh:.4f}")
                if self.tracker is not None:
                    self.tracker.log(f"{name}.q_low", ql, stage="calibrate")
                    self.tracker.log(f"{name}.q_high", qh, stage="calibrate")

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

        results = evaluate_on_test_set(
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
                            f"{name}.{metric_name}",
                            float(value),
                            stage="evaluate",
                        )
                for metric_name, value in cal.items():
                    if isinstance(value, (int, float)):
                        self.tracker.log(
                            f"{name}.calib_{metric_name}",
                            float(value),
                            stage="evaluate",
                        )

        return results

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
