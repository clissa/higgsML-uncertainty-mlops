"""Reusable training orchestrator for the toy conformal-prediction pipeline.

The ``Trainer`` class encapsulates the full train → evaluate →
nonconformity → confidence-interval flow that was previously baked into
``scripts/train.py#main()``.  Scientific computation is delegated to
the pure functions in ``conformal_predictions.training.core``; the
``Trainer`` is a *thin* stateful wrapper that wires everything together
and writes artifacts into the run-context output directory.

Usage::

    from conformal_predictions.config import load_training_config
    from conformal_predictions.mlops.run_context import RunContext
    from conformal_predictions.training.trainer import Trainer

    cfg = load_training_config("configs/train_toy.yaml")
    ctx = RunContext.create(cfg, config_path="configs/train_toy.yaml")
    trainer = Trainer(cfg, ctx)
    trainer.run()

TODO Phase 1b: Extend to Higgs pipeline via a pluggable data-loader
interface (load_data returns the same structured tuple; only the
reading logic differs).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from conformal_predictions.config import TrainingConfig
from conformal_predictions.data.toy import load_pseudo_experiment
from conformal_predictions.data_viz import (
    contourplot_data,
    plot_confidence_intervals,
    plot_mu_hat_distribution,
    plot_nonconformity_scores,
)
from conformal_predictions.mlops.run_context import RunContext
from conformal_predictions.training.core import (
    compute_confidence_interval,
    compute_mu_hat,
    compute_nonconformity_scores,
    evaluate_models,
    get_events_count,
    inference_on_test_set,
    list_split_files,
)
from conformal_predictions.training.models import build_default_models


class Trainer:
    """Config-driven training orchestrator (toy pipeline).

    Parameters
    ----------
    config : TrainingConfig
        Frozen training configuration.
    run_ctx : RunContext
        Run metadata & output directories.
    """

    def __init__(self, config: TrainingConfig, run_ctx: RunContext) -> None:
        self.config = config
        self.run_ctx = run_ctx
        self.models: Dict[str, object] = {}
        self.scaler: Optional[StandardScaler] = None

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
    # Full pipeline
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the full train → eval → nonconformity → CI pipeline."""
        cfg = self.config
        ctx = self.run_ctx
        ctx.ensure_dirs()

        np.random.seed(cfg.seed)

        # ---- data ----
        X_train, y_train, X_val, y_val, calib_data, calib_meta, test_files = (
            self.load_data()
        )

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

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

        # ---- evaluation ----
        perf = evaluate_models(self.models, X_val_scaled, y_val)
        for name, m in perf.items():
            print(f"\n\n{name} validation metrics:")
            print(
                f"\tAccuracy: {m['accuracy']:.4f}"
                f"\tPrecision: {m['precision']:.4f}"
                f"\tRecall: {m['recall']:.4f}"
                f"\tF1: {m['f1']:.4f}"
            )

        counts = get_events_count(self.models, X_val_scaled, cfg.threshold)
        for name, count in counts.items():
            print(
                f"{name} N signal events (p_pred > {cfg.threshold}): "
                f"{count} / {int(np.sum(y_val))} (true)"
            )

        # ---- nonconformity scores ----
        print("\nComputing nonconformity scores...")
        print(f"{len(calib_data)} calibration samples")
        print(
            f"Average calibration sample size: "
            f"{int(np.array([c[0].shape[0] for c in calib_data]).mean())} observations"
        )
        print(
            f"\t...using {cfg.nonconf_target} as target for nonconformity scores"
        )
        nonconf_scores = compute_nonconformity_scores(
            self.models,
            self.scaler,
            calib_data,
            calib_meta,
            cfg.threshold,
            target=cfg.nonconf_target,
            how=cfg.nonconf_method,
        )

        for name, values in nonconf_scores.items():
            mu = np.mean(values) if values else float("nan")
            sd = np.std(values) if values else float("nan")
            print(f"{name} nonconformity mu_hat stats: {mu:.4f} ± {sd:.4f}")

        plot_nonconformity_scores(
            nonconf_scores,
            scores_label=cfg.nonconf_target,
            output_dir=ctx.plots_dir,
        )

        # ---- mu_hat on calibration set ----
        print("\nComputing mu_hat...")
        mu_hat, stats = compute_mu_hat(
            self.models, self.scaler, calib_data, calib_meta, cfg.threshold
        )
        np.savez(
            ctx.stats_dir / "mu_hat_calib_distribution.npz",
            **{n: np.array(v) for n, v in mu_hat.items()},
        )
        np.savez(
            ctx.stats_dir / f"{cfg.nonconf_target}_nonconf_scores.npz",
            **{n: np.array(v) for n, v in nonconf_scores.items()},
        )

        plot_mu_hat_distribution(mu_hat, stats, output_dir=ctx.plots_dir)
        df_stats = pd.DataFrame(
            [{"Model": n, **stats[n]} for n in stats]
        )
        print("\nStatistics Summary:")
        print(df_stats)
        df_stats.to_csv(ctx.stats_dir / "mu_hat_calibration_stats.csv", index=False)

        # ---- inference on test set ----
        print("\nRunning inference on test set...")
        test_data = []
        for fp in test_files:
            X_t, y_t, meta_t = load_pseudo_experiment(fp)
            test_data.append((X_t, y_t, meta_t))

        mu_hat_test, mu_true_list, gamma_true_list, test_metrics = (
            inference_on_test_set(self.models, self.scaler, test_data, cfg.threshold)
        )

        # ---- confidence intervals ----
        print("\nComputing confidence intervals for test set...")
        print(
            f"\t...using {cfg.nonconf_target} nonconformity scores for CI computation."
        )
        nonconf_scores_file = (
            ctx.stats_dir / f"{cfg.nonconf_target}_nonconf_scores.npz"
        )

        for model_name, mu_hat_values in mu_hat_test.items():
            if cfg.nonconf_target == "mu_hat":
                lower, upper = compute_confidence_interval(
                    np.array(mu_hat_values),
                    nonconf_scores_file,
                    model_name,
                    how=cfg.nonconf_method,
                )
            elif cfg.nonconf_target == "n_pred":
                n_preds = np.array(mu_hat_values) * np.array(gamma_true_list)
                n_lower, n_upper = compute_confidence_interval(
                    n_preds,
                    nonconf_scores_file,
                    model_name,
                    how=cfg.nonconf_method,
                )
                lower = n_lower / np.array(gamma_true_list)
                upper = n_upper / np.array(gamma_true_list)

            print(f"\nModel: {model_name}\n")
            empirical_coverage = np.mean(
                [
                    lo < mt < hi
                    for lo, hi, mt in zip(lower, upper, mu_true_list)
                ]
            )
            print(f"Empirical coverage: {empirical_coverage*100:.2f}%")

            plot_confidence_intervals(
                mu_hat_values,
                lower,
                upper,
                mu_true_list,
                model_name,
                empirical_coverage,
                output_dir=ctx.stats_dir,
            )

            for i, (mh, lo, hi, mt) in enumerate(
                zip(mu_hat_values, lower, upper, mu_true_list)
            ):
                if i % 50 != 0:
                    continue
                color = "\033[92m" if lo < mt < hi else "\033[91m"
                reset = "\033[0m"
                print(
                    f"  {color}Exp {i}: "
                    f"\u03bc\u0302: {mh:.3f} "
                    f"CI: [{lo:.3f}, {hi:.3f}] "
                    f"\u03bc_true: {mt:.3f}{reset}"
                )
                print(test_metrics[model_name][i])

        # ---- persist run metadata ----
        meta_path = ctx.save_metadata()
        print(f"\nRun metadata saved to {meta_path}")
