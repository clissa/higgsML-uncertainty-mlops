from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from conformal_predictions.data.toy import load_pseudo_experiment

# TODO: Refactor to support yaml config loading. It should take Settings attributes + OUTPUT_DIRNAME. Do not change parts/names that are not necessary for this.
OUTPUT_DIRNAME = "toy-scale-95"
PLOTS_DIR = Path("results") / "plots" / OUTPUT_DIRNAME
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

STATS_DIR = Path("results") / "stats" / OUTPUT_DIRNAME
STATS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data") / "toy_scale"
    mu: float = 1.0
    seed: int = 18
    test_prefixes: Tuple[str, ...] = ("7e39", "6fcb")
    threshold: float = 0.5
    valid_size: float = 0.2
    calib_size: float = 0.3


def _experiment_prefix(path: Path) -> str:
    stem = path.stem
    if stem.startswith("experiment_"):
        stem = stem[len("experiment_") :]
    return stem[:4]


def _list_split_files(
    data_dir: Path, mu: float, test_prefixes: Sequence[str]
) -> Tuple[List[Path], List[Path]]:
    mu_dir = data_dir / f"mu={mu}"
    files = sorted(mu_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {mu_dir}")
    test_files = [path for path in files if _experiment_prefix(path) in test_prefixes]
    if not test_files:
        print(
            "No test files found with the specified prefixes. Falling back to 5 random files."
        )
        test_files = list(np.random.choice(files, size=5, replace=False))
    train_files = [path for path in files if path not in test_files]
    if not train_files:
        raise ValueError("No training files remain after test split.")
    return train_files, test_files


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
    for model in models.values():
        model.fit(X_train, y_train)


# TODO: Add ROC AUC scoring as well. Important for linking model bias to Precision-Recall
def _score_models(
    models: Dict[str, object], X_val: np.ndarray, y_val: np.ndarray
) -> Dict[str, float]:
    return {name: float(model.score(X_val, y_val)) for name, model in models.items()}


def _get_events_count(
    models: Dict[str, object], X_val: np.ndarray, threshold: float
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        counts[name] = int(np.sum(y_pred_proba > threshold))
    return counts


def _nonconformity_scores(
    models: Dict[str, object],
    scaler: StandardScaler,
    calib_data: Sequence[Tuple[np.ndarray, np.ndarray]],
    threshold: float,
) -> Dict[str, List[int]]:
    scores: Dict[str, List[int]] = {name: [] for name in models}
    for X_calib, y_calib in calib_data:
        X_calib = scaler.transform(X_calib)
        n_obs = int(np.sum(y_calib))
        for name, model in models.items():
            y_pred_proba = model.predict_proba(X_calib)[:, 1]
            n_pred = int(np.sum(y_pred_proba > threshold))
            # TODO: USE n_obs - n_pred instead for more intuitive CI computation!
            scores[name].append(n_pred - n_obs)
    return scores


def _compute_mu_hat(
    models: Dict[str, object],
    scaler: StandardScaler,
    calib_data: Sequence[Tuple[np.ndarray, np.ndarray]],
    threshold: float,
) -> Tuple[Dict[str, List[float]], Dict[str, Dict[str, float]]]:
    mu_hat: Dict[str, List[float]] = {name: [] for name in models}
    for X_calib, y_calib in calib_data:
        X_calib = scaler.transform(X_calib)
        # TODO: fix using gamma_true from metadata!
        gamma_true = int(np.sum(y_calib))
        if gamma_true == 0:
            continue
        for name, model in models.items():
            y_pred_proba = model.predict_proba(X_calib)[:, 1]
            n_pred = int(np.sum(y_pred_proba > threshold))
            mu_hat[name].append(n_pred / gamma_true)

    # Compute statistics
    stats: Dict[str, Dict[str, float]] = {}
    for name, values in mu_hat.items():
        if len(values) > 0:
            # Compute KDE for MAP estimation
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


def plot_mu_hat_distribution(
    mu_hat: Dict[str, List[float]],
    stats: Dict[str, Dict[str, float]],
    output_dir: Path = Path("plots"),
) -> None:
    """Plot mu_hat distributions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = list(mu_hat.keys())

    for model_name in model_names:
        values = mu_hat[model_name]
        if len(values) == 0:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(values, bins=30, alpha=0.7, color="blue", density=True)

        if len(values) > 1:
            density = gaussian_kde(values)
            xs = np.linspace(min(values), max(values), 200)
            ax.plot(xs, density(xs), "k-", linewidth=2, label="KDE")

        s = stats[model_name]
        ax.axvline(
            s["q16"],
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"q16: {s['q16']:.3f}",
        )
        ax.axvline(
            s["mu_median"],
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"median: {s['mu_median']:.3f}",
        )
        ax.axvline(
            s["mu_mean"],
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"mean: {s['mu_mean']:.3f}",
        )
        ax.axvline(
            s["q84"],
            color="purple",
            linestyle="--",
            linewidth=2,
            label=f"q84: {s['q84']:.3f}",
        )
        ax.axvline(
            s["map"],
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"MAP: {s['map']:.3f}",
        )

        ax.set_title(f"μ̂ Distribution: {model_name}")
        ax.set_xlabel("μ̂ = n_pred / γ_true")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            output_dir / f"mu_hat_distribution_{model_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def plot_nonconformity_scores(
    nonconf_scores: Dict[str, List[int]], output_dir: Path = Path("plots")
) -> None:
    """Plot nonconformity score distributions for each model and comparison."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = list(nonconf_scores.keys())

    for model_name in model_names:
        fig, ax = plt.subplots(figsize=(10, 6))
        scores = nonconf_scores[model_name]

        ax.hist(scores, bins=30, alpha=0.7, color="blue", density=True)

        if len(scores) > 1:
            density = gaussian_kde(scores)
            xs = np.linspace(min(scores), max(scores), 200)
            ax.plot(xs, density(xs), "k-", linewidth=2, label="KDE")

        ax.set_title(f"Nonconformity Scores Distribution: {model_name}")
        ax.set_xlabel("Nonconformity Score")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            output_dir / f"scores_distribution_{model_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    fig, axs = plt.subplots(len(model_names), 1, figsize=(10, 5 * len(model_names)))
    if len(model_names) == 1:
        axs = [axs]

    colors = ["blue", "green", "red"]
    for i, model_name in enumerate(model_names):
        scores = nonconf_scores[model_name]
        bin_width = 1
        axs[i].hist(
            scores,
            bins=np.arange(min(scores), max(scores) + bin_width, bin_width),
            alpha=0.7,
            color=colors[i % len(colors)],
            density=True,
        )

        if len(scores) > 1:
            density = gaussian_kde(scores)
            xs = np.linspace(min(scores), max(scores), 200)
            axs[i].plot(xs, density(xs), "k-", linewidth=2, label="KDE")

        axs[i].set_title(f"{model_name}")
        axs[i].set_xlabel("Nonconformity Score")
        axs[i].set_ylabel("Density")
        axs[i].legend()
        axs[i].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "scores_distribution_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def contourplot_data(
    X: np.ndarray, y: np.ndarray, output_dir: Path = Path("plots")
) -> None:
    """
    Plot 2D density contours with overlaid scatter points for
    background (y=0) and signal (y=1).

    Background is drawn first, then signal.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if X.shape[1] < 2:
        return
    signal = X[y == 1][:, :2]
    background = X[y == 0][:, :2]
    if len(signal) == 0 or len(background) == 0:
        return
    fig, ax = plt.subplots(figsize=(9, 7))
    # Plot limits with small margins
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    dx = 0.05 * (x_max - x_min)
    dy = 0.05 * (y_max - y_min)
    x_grid, y_grid = np.mgrid[
        (x_min - dx) : (x_max + dx) : 200j,
        (y_min - dy) : (y_max + dy) : 200j,
    ]
    grid = np.vstack([x_grid.ravel(), y_grid.ravel()])
    # KDEs
    background_kde = gaussian_kde(background.T)
    signal_kde = gaussian_kde(signal.T)
    background_density = background_kde(grid).reshape(x_grid.shape)
    signal_density = signal_kde(grid).reshape(x_grid.shape)
    # --- Background ---
    ax.contourf(
        x_grid,
        y_grid,
        background_density,
        levels=10,
        cmap="Blues",
        alpha=0.6,
    )
    ax.contour(
        x_grid,
        y_grid,
        background_density,
        levels=6,
        colors="blue",
        linewidths=1.0,
    )
    ax.scatter(
        background[:, 0],
        background[:, 1],
        s=8,
        c="blue",
        alpha=0.15,
        label="Background",
        rasterized=True,
    )
    # --- Signal ---
    ax.contourf(
        x_grid,
        y_grid,
        signal_density,
        levels=10,
        cmap="Reds",
        alpha=0.6,
    )
    ax.contour(
        x_grid,
        y_grid,
        signal_density,
        levels=6,
        colors="red",
        linewidths=1.2,
    )
    ax.scatter(
        signal[:, 0],
        signal[:, 1],
        s=10,
        c="red",
        alpha=0.25,
        label="Signal",
        rasterized=True,
    )
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
    ax.set_title("Signal vs Background Density")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "data_contour.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _split_train_val_calib(
    X: np.ndarray,
    y: np.ndarray,
    valid_size: float,
    calib_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if valid_size < 0 or calib_size < 0 or valid_size + calib_size > 0.5:
        raise ValueError("valid_size and calib_size must be >= 0 and sum to <= 0.5.")
    X_temp, X_calib, y_temp, y_calib = train_test_split(
        X, y, test_size=calib_size, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=valid_size, stratify=y_temp, random_state=seed
    )
    return X_train, X_val, X_calib, y_train, y_val, y_calib


def _inference_on_test_set(
    models: Dict[str, object],
    scaler: StandardScaler,
    test_data: Sequence[Tuple[np.ndarray, np.ndarray, dict]],
    threshold: float,
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
    mu_true_list: List[float] = []
    gamma_true_list: List[int] = []

    for X_test, y_test, meta_dict in test_data:
        # Transform features
        X_test_scaled = scaler.transform(X_test)

        # Get gamma_true from metadata
        gamma_true = meta_dict["gamma_true"]
        mu_true = meta_dict["mu_true"]
        mu_true_list.append(float(mu_true))
        gamma_true_list.append(int(gamma_true))

        if gamma_true == 0:
            continue

        # Compute predictions for each model
        for name, model in models.items():
            # Get probability predictions for signal class
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

            # Count predicted signal events
            n_pred = int(np.sum(y_pred_proba > threshold))

            # Compute mu_hat
            mu_hat = n_pred / gamma_true
            mu_hat_test[name].append(mu_hat)

            # Debug prints
            print("\nDebug prints:", name)
            n_obs = int(np.sum(y_test))
            print(
                f"\tExperiment: mu_true={mu_true:.4f}, gamma_true={gamma_true}, n_obs={n_obs}, n_pred={n_pred}, mu_hat={mu_hat:.4f}"
            )

    return mu_hat_test, mu_true_list, gamma_true_list


def compute_confidence_interval(
    n_pred: int,
    nonconf_scores_file: Path,
    model_name: str,
) -> Tuple[float, float]:
    """
    Compute confidence interval from calibration nonconformity scores.

    Args:
        n_pred: Number of predicted signal events
        nonconf_scores_file: Path to .npz file containing nonconformity scores
        model_name: Name of the model to extract scores for

    Returns:
        Tuple of (lower_bound, upper_bound) for the confidence interval
    """
    data = np.load(nonconf_scores_file)
    if model_name not in data:
        raise KeyError(f"Model '{model_name}' not found in {nonconf_scores_file}")

    scores = data[model_name]
    q16 = float(np.percentile(scores, 16))
    q84 = float(np.percentile(scores, 84))

    lower_bound = n_pred - q84
    upper_bound = n_pred - q16

    return lower_bound, upper_bound


def main() -> None:
    cfg = Settings()
    np.random.seed(cfg.seed)

    train_files, _test_files = _list_split_files(
        cfg.data_dir, cfg.mu, cfg.test_prefixes
    )
    train_blocks: List[np.ndarray] = []
    val_blocks: List[np.ndarray] = []
    train_labels: List[np.ndarray] = []
    val_labels: List[np.ndarray] = []
    calib_data: List[Tuple[np.ndarray, np.ndarray]] = []
    for file_path in train_files:
        X, y, _meta = load_pseudo_experiment(file_path)
        (
            X_train,
            X_val,
            X_calib,
            y_train,
            y_val,
            y_calib,
        ) = _split_train_val_calib(X, y, cfg.valid_size, cfg.calib_size, cfg.seed)
        train_blocks.append(X_train)
        val_blocks.append(X_val)
        train_labels.append(y_train)
        val_labels.append(y_val)
        calib_data.append((X_calib, y_calib))

    X_train = np.vstack(train_blocks)
    X_val = np.vstack(val_blocks)
    y_train = np.concatenate(train_labels)
    y_val = np.concatenate(val_labels)

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

    contourplot_data(X_val, y_val, output_dir=PLOTS_DIR)

    models = _build_models(cfg.seed)
    _fit_models(models, X_train_scaled, y_train)

    scores = _score_models(models, X_val_scaled, y_val)
    for model_name, score in scores.items():
        print(f"{model_name} validation accuracy: {score:.4f}")

    counts = _get_events_count(models, X_val_scaled, cfg.threshold)
    for model_name, count in counts.items():
        print(
            f"{model_name} N signal events (p_pred > {cfg.threshold}): "
            f"{count} / {int(np.sum(y_val))} (true)"
        )

    print("\nComputing nonconformity scores...")
    print(f"{len(calib_data)} calibration samples")
    print(
        f"Average calibration sample size: {int(np.array([_[0].shape[0] for _ in calib_data]).mean())} observations"
    )

    nonconf_scores = _nonconformity_scores(models, scaler, calib_data, cfg.threshold)
    for model_name, values in nonconf_scores.items():
        mean_score = float(np.mean(values)) if values else float("nan")
        print(f"{model_name} nonconformity mean: {mean_score:.4f}")

    plot_nonconformity_scores(nonconf_scores, output_dir=PLOTS_DIR)
    print("\nComputing mu_hat...")
    mu_hat, stats = _compute_mu_hat(models, scaler, calib_data, cfg.threshold)
    np.savez(
        STATS_DIR / "mu_hat_nonconf_scores.npz",
        **{model_name: np.array(scores) for model_name, scores in mu_hat.items()},
    )
    np.savez(
        STATS_DIR / "n_pred_nonconf_scores.npz",
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
    test_data = []
    for file_path in _test_files:
        X_test, y_test, meta = load_pseudo_experiment(file_path)
        test_data.append((X_test, y_test, meta))

    mu_hat_test, mu_true_list, gamma_true_list = _inference_on_test_set(
        models, scaler, test_data, cfg.threshold
    )

    print("\nTest set mu_hat estimates:")
    for model_name, values in mu_hat_test.items():
        print(f"  {model_name}: values={values}")
        print(f"  {model_name}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")
    print(f"  mu_true values: {mu_true_list}")
    print(f"  gamma_true values: {gamma_true_list}")

    # Compute confidence intervals for test set predictions
    print("\nComputing confidence intervals for test set...")

    # Load nonconformity scores for computing intervals
    # nonconf_scores_file = STATS_DIR / "mu_hat_nonconf_scores.npz"
    nonconf_scores_file = STATS_DIR / "n_pred_nonconf_scores.npz"

    for i, (model_name, mu_hat_values) in enumerate(mu_hat_test.items()):
        n_preds = mu_hat_values * np.array(gamma_true_list)
        n_lower, n_upper = compute_confidence_interval(
            n_preds, nonconf_scores_file, model_name
        )

        mu_hat_lower_bounds = n_lower / np.array(gamma_true_list)
        mu_hat_upper_bounds = n_upper / np.array(gamma_true_list)
        print(f"\nModel: {model_name}\n")
        exp_idx = 1
        for mu_hat, mu_hat_lower, mu_hat_upper, mu_true in zip(
            mu_hat_values, mu_hat_lower_bounds, mu_hat_upper_bounds, mu_true_list
        ):
            # Determine color based on whether CI contains mu_true
            color = "\033[92m" if mu_hat_lower < mu_true < mu_hat_upper else "\033[91m"
            reset = "\033[0m"
            print(
                f"  {color}Exp {exp_idx}: "
                f"μ̂: {mu_hat:.3f} "
                f"CI: [{mu_hat_lower:.3f}, {mu_hat_upper:.3f}] "
                f"μ_true: {mu_true:.3f}{reset}"
            )
            exp_idx += 1


if __name__ == "__main__":
    main()
