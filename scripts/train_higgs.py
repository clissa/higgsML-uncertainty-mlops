from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from conformal_predictions.data_viz import (
    contourplot_data,
    plot_confidence_intervals,
    plot_mu_hat_distribution,
    plot_nonconformity_scores,
)
from conformal_predictions.training import (
    compute_confidence_interval,
    compute_mu_hat,
    compute_nonconformity_scores,
    evaluate_models,
    get_events_count,
    inference_on_test_set,
)

# TODO: Refactor to support yaml config loading. It should take Settings attributes + OUTPUT_DIRNAME. Do not change parts/names that are not necessary for this.
OUTPUT_DIRNAME = "higgs-2train-1valid-1calib-1test"
PLOTS_DIR = Path("results") / OUTPUT_DIRNAME / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

STATS_DIR = Path("results") / OUTPUT_DIRNAME / "stats"
STATS_DIR.mkdir(parents=True, exist_ok=True)

# TODO: Refactor to support yaml config loading. It should take Settings attributes + OUTPUT_DIRNAME. Do not change parts/names that are not necessary for this.


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data") / "HiggsML/input_data/train"
    mu: float = 1.0
    seed: int = 18
    # test_prefixes: Tuple[str, ...] = ("7e39", "6fcb")
    threshold: float = 0.5
    train_size: float = 10
    valid_size: float = 10
    calib_size: float = 10
    test_size: float = 10
    nonconf_target: str = "mu_hat"  # can be "n_pred" or "mu_hat"
    block_size: int = (
        10_000  # number of test pseudo-experiments to select if no prefixes match
    )


def load_trainval(
    cfg: Settings,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read:
      - data/data.parquet
      - labels/data.labels

    Use:
      - row groups 0,1, ...train_size -> train
      - row group train_size, ..., train_size + valid_size -> validation
    """
    base_dir = cfg.data_dir
    parquet_path = base_dir / "data" / "data.parquet"
    labels_path = base_dir / "labels" / "data.labels"
    # --- Read parquet ---
    pf = pq.ParquetFile(parquet_path)
    # first 3 row groups
    # train_size = 2
    # valid_size = 1
    train_tables = [pf.read_row_group(i) for i in range(cfg.train_size)]
    val_table = [
        pf.read_row_group(i)
        for i in range(cfg.train_size, cfg.train_size + cfg.valid_size)
    ]
    # convert to numpy
    X_train = np.vstack([t.to_pandas().to_numpy() for t in train_tables])
    X_val = np.vstack([t.to_pandas().to_numpy() for t in val_table])
    # read labels
    y_all = np.loadtxt(labels_path)
    # numero righe per ciascun row group
    # rg0_rows = pf.metadata.row_group(0).num_rows
    # rg1_rows = pf.metadata.row_group(1).num_rows
    # rg2_rows = pf.metadata.row_group(2).num_rows
    # slicing labels
    y_train = y_all[: X_train.shape[0]]
    y_val = y_all[X_train.shape[0] : X_train.shape[0] + X_val.shape[0]]
    return X_train, y_train, X_val, y_val


def load_calib(
    cfg: Settings, calib_start_label_idx: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Legge:
      - data/data.parquet
      - labels/data.labels

    Usa:
      - row groups 0,1, ...train_size -> train
      - row group train_size, ..., train_size + valid_size -> validation
    """
    base_dir = cfg.data_dir
    parquet_path = base_dir / "data" / "data.parquet"
    labels_path = base_dir / "labels" / "data.labels"
    # --- Leggi parquet ---
    pf = pq.ParquetFile(parquet_path)
    # primi 3 row groups
    # train_size = 2
    # valid_size = 1
    # calib_size = 2
    calib_start_idx = cfg.train_size + cfg.valid_size
    calib_tables = [
        pf.read_row_group(i)
        for i in range(calib_start_idx, calib_start_idx + cfg.calib_size)
    ]
    # convert to numpy
    X_calib = np.vstack([t.to_pandas().to_numpy() for t in calib_tables])
    # read labels
    y_all = np.loadtxt(labels_path)
    y_calib = y_all[calib_start_label_idx : calib_start_label_idx + X_calib.shape[0]]
    # X_calib_blocks = []
    # y_calib_blocks = []
    calib_data = []
    metadata = []
    for i in range(0, X_calib.shape[0], cfg.block_size):
        # X_calib_blocks.append(X_calib[i : i + cfg.block_size])
        # y_calib_blocks.append(y_calib[i : i + cfg.block_size])
        block_length = X_calib[i : i + cfg.block_size].shape[0]
        calib_data.append(
            (X_calib[i : i + cfg.block_size], y_calib[i : i + cfg.block_size])
        )
        gamma_true = 336.0 * (
            block_length / 1000
        )  # scale gamma_true proportionally to block size
        meta_dict = {
            "mu_true": 1.0,
            "gamma_true": gamma_true,
            "beta_true": block_length - gamma_true,
            "nu_expected": block_length,
            "n_signal": y_calib[i : i + cfg.block_size].sum(),
            "n_background": (1 - y_calib[i : i + cfg.block_size]).sum(),
            "n_total": block_length,
            # "weights": weights,
        }
        metadata.append(meta_dict)
    print(
        f"Calibration sample has: {X_calib.shape[0]} events and {y_calib.shape[0]} labels."
    )
    print(
        f"Calibration events are splitted into {len(calib_data)} blocks of size {cfg.block_size} (last block may be smaller)"
    )
    return calib_data, metadata


def load_test(
    cfg: Settings, test_start_label_idx: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Legge:
      - data/data.parquet
      - labels/data.labels

    Usa:
      - row groups 0,1, ...train_size -> train
      - row group train_size, ..., train_size + valid_size -> validation
    """
    base_dir = cfg.data_dir
    parquet_path = base_dir / "data" / "data.parquet"
    labels_path = base_dir / "labels" / "data.labels"
    # --- Leggi parquet ---
    pf = pq.ParquetFile(parquet_path)
    # primi 3 row groups
    # train_size = 2
    # valid_size = 1
    # calib_size = 2
    # test_size = 1
    test_start_idx = cfg.train_size + cfg.valid_size + cfg.calib_size
    test_tables = [
        pf.read_row_group(i)
        for i in range(test_start_idx, test_start_idx + cfg.test_size)
    ]
    # convert to numpy
    X_test = np.vstack([t.to_pandas().to_numpy() for t in test_tables])
    # read labels
    y_all = np.loadtxt(labels_path)
    y_test = y_all[test_start_label_idx : test_start_label_idx + X_test.shape[0]]
    test_data = []
    for i in range(0, X_test.shape[0], cfg.block_size):
        _X_test_blocks = X_test[i : i + cfg.block_size]
        _y_test_blocks = y_test[i : i + cfg.block_size]
        block_length = X_test[i : i + cfg.block_size].shape[0]
        gamma_true = 336.0 * (
            block_length / 1000
        )  # scale gamma_true proportionally to block size
        _meta_dict = {
            "mu_true": 1.0,
            "gamma_true": gamma_true,
            "beta_true": block_length - gamma_true,
            "nu_expected": block_length,
            "n_signal": y_test[i : i + cfg.block_size].sum(),
            "n_background": (1 - y_test[i : i + cfg.block_size]).sum(),
            "n_total": block_length,
            # "weights": weights,
        }
        test_data.append([_X_test_blocks, _y_test_blocks, _meta_dict])
    print(f"Test sample has: {X_test.shape[0]} events and {y_test.shape[0]} labels.")
    print(
        f"Test events are splitted into {len(test_data)} blocks of size {cfg.block_size} (last block may be smaller)"
    )
    return test_data


# TODO: Add more models and hyperparameter tuning: in particular, try probability regression VS classification.
def _build_models(seed: int) -> Dict[str, object]:
    return {
        "GLM": LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            random_state=seed,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=50,
            criterion="gini",
            n_jobs=-1,
            random_state=seed,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            max_iter=1000,
            random_state=seed,
        ),
    }


def _fit_models(
    models: Dict[str, object], X_train: np.ndarray, y_train: np.ndarray
) -> None:
    for model in tqdm(models.values(), desc="Training models"):
        model.fit(X_train, y_train)


def main() -> None:
    cfg = Settings()
    np.random.seed(cfg.seed)

    X_train, y_train, X_val, y_val = load_trainval(cfg)
    n_trainval = X_train.shape[0] + X_val.shape[0]
    # X_train, y_train, X_val, y_val = (
    #     X_train[:10000],
    #     y_train[:10000],
    #     X_val[:5000],
    #     y_val[:5000],
    # )
    calib_data, calib_meta = load_calib(cfg, calib_start_label_idx=n_trainval)
    n_calib = np.sum([_[0].shape[0] for _ in calib_data])
    # calib_data, calib_meta = calib_data[:100], calib_meta[:100]
    test_data = load_test(cfg, test_start_label_idx=n_trainval + n_calib)
    n_test = np.sum([_[0].shape[0] for _ in test_data])
    # test_data = test_data[:100]
    print(f"Total train+val events: {n_trainval}")
    print(f"Total calib events: {n_calib}\t{len(calib_data)} blocks")
    print(f"Total test events: {n_test}\t{len(test_data)} blocks")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print("Starting training on:")
    print(f"  X_train shape: {X_train_scaled.shape}, y_train shape: {y_train.shape}")
    print(f"  X_val shape: {X_val_scaled.shape}, y_val shape: {y_val.shape}")
    print(
        f"  y_train: {int(np.sum(y_train))} positives "
        f"({100*np.sum(y_train)/len(y_train):.1f}%), "
        f"{len(y_train) - int(np.sum(y_train))} negatives "
        f"({100*(len(y_train)-np.sum(y_train))/len(y_train):.1f}%)"
    )
    print(
        f"  y_val: {int(np.sum(y_val))} positives "
        f"({100*np.sum(y_val)/len(y_val):.1f}%), "
        f"{len(y_val) - int(np.sum(y_val))} negatives "
        f"({100*(len(y_val)-np.sum(y_val))/len(y_val):.1f}%)"
    )

    contourplot_data(X_val_scaled, y_val, output_dir=PLOTS_DIR)

    models = _build_models(cfg.seed)
    _fit_models(models, X_train_scaled, y_train)

    # print classification performance on validation set
    performance_metrics = evaluate_models(models, X_val_scaled, y_val)
    for model_name, metrics in performance_metrics.items():
        print(f"\n\n{model_name} validation metrics:")
        print(
            f"\tAccuracy: {metrics['accuracy']:.4f}"
            f"\tPrecision: {metrics['precision']:.4f}"
            f"\tRecall: {metrics['recall']:.4f}"
            f"\tF1: {metrics['f1']:.4f}"
        )

    counts = get_events_count(models, X_val_scaled, cfg.threshold)
    for model_name, count in counts.items():
        print(
            f"{model_name} N signal events (p_pred > {cfg.threshold}): "
            f"{count} / {int(np.sum(y_val))} (true)"
        )

    # nonconformity scores
    print("\nComputing nonconformity scores...")
    print(f"{len(calib_data)} calibration samples")
    print(
        f"Average calibration sample size: {int(np.array([_[0].shape[0] for _ in calib_data]).mean())} observations"
    )
    print(f"\t...using {cfg.nonconf_target} as target for nonconformity scores")
    nonconf_scores = compute_nonconformity_scores(
        models, scaler, calib_data, calib_meta, cfg.threshold, target=cfg.nonconf_target
    )

    for model_name, values in nonconf_scores.items():
        mean_score = np.mean(values) if values else float("nan")
        std_score = np.std(values) if values else float("nan")
        print(
            f"{model_name} nonconformity mu_hat stats: {mean_score:.4f} ± {std_score:.4f}"
        )

    plot_nonconformity_scores(
        nonconf_scores, scores_label=cfg.nonconf_target, output_dir=PLOTS_DIR
    )

    print("\nComputing mu_hat...")
    mu_hat, stats = compute_mu_hat(
        models, scaler, calib_data, calib_meta, cfg.threshold
    )
    np.savez(
        STATS_DIR / "mu_hat_calib_distribution.npz",
        **{model_name: np.array(scores) for model_name, scores in mu_hat.items()},
    )
    np.savez(
        STATS_DIR / f"{cfg.nonconf_target}_nonconf_scores.npz",
        **{
            model_name: np.array(scores)
            for model_name, scores in nonconf_scores.items()
        },
    )

    plot_mu_hat_distribution(mu_hat, stats, output_dir=PLOTS_DIR)
    df_stats = pd.DataFrame(
        [{"Model": model_name, **stats[model_name]} for model_name in stats.keys()]
    )
    print("\nStatistics Summary:")
    print(df_stats)
    df_stats.to_csv(STATS_DIR / "mu_hat_calibration_stats.csv", index=False)

    print("\nRunning inference on test set...")

    mu_hat_test, mu_true_list, gamma_true_list, test_metrics = inference_on_test_set(
        models, scaler, test_data, cfg.threshold
    )

    # Compute confidence intervals for test set predictions
    print("\nComputing confidence intervals for test set...")

    # Load nonconformity scores for computing intervals
    print(f"\t...using {cfg.nonconf_target} nonconformity scores for CI computation.")
    nonconf_scores_file = STATS_DIR / f"{cfg.nonconf_target}_nonconf_scores.npz"

    for model_name, mu_hat_values in mu_hat_test.items():
        if cfg.nonconf_target == "mu_hat":
            mu_hat_lower_bounds, mu_hat_upper_bounds = compute_confidence_interval(
                np.array(mu_hat_values), nonconf_scores_file, model_name
            )
        elif cfg.nonconf_target == "n_pred":
            n_preds = mu_hat_values * np.array(gamma_true_list)
            n_lower, n_upper = compute_confidence_interval(
                n_preds, nonconf_scores_file, model_name
            )
            mu_hat_lower_bounds = n_lower / np.array(gamma_true_list)
            mu_hat_upper_bounds = n_upper / np.array(gamma_true_list)
        print(f"\nModel: {model_name}\n")
        empirical_coverage = np.mean(
            [
                mu_hat_lower < mu_true < mu_hat_upper
                for mu_hat_lower, mu_hat_upper, mu_true in zip(
                    mu_hat_lower_bounds,
                    mu_hat_upper_bounds,
                    mu_true_list,
                )
            ]
        )
        print(f"Empirical coverage: {empirical_coverage*100:.2f}%")
        plot_confidence_intervals(
            mu_hat_values,
            mu_hat_lower_bounds,
            mu_hat_upper_bounds,
            mu_true_list,
            model_name,
            empirical_coverage,
            output_dir=STATS_DIR,
        )
        for exp_idx, (mu_hat, mu_hat_lower, mu_hat_upper, mu_true) in enumerate(
            zip(mu_hat_values, mu_hat_lower_bounds, mu_hat_upper_bounds, mu_true_list)
        ):
            if exp_idx % 50 != 0:
                # exp_idx += 1
                continue
            print("printing every 50th experiment:")
            # Determine color based on whether CI contains mu_true
            color = "\033[92m" if mu_hat_lower < mu_true < mu_hat_upper else "\033[91m"
            reset = "\033[0m"
            print(
                f"  {color}Exp {exp_idx}: "
                f"μ̂: {mu_hat:.3f} "
                f"CI: [{mu_hat_lower:.3f}, {mu_hat_upper:.3f}] "
                f"μ_true: {mu_true:.3f}{reset}"
            )
            print(test_metrics[model_name][exp_idx])


if __name__ == "__main__":
    main()
