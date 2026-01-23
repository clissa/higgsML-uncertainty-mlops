from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


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
    nonconf_scores: Dict[str, List[int]],
    scores_label: str,
    output_dir: Path = Path("plots"),
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
        ax.set_xlabel(f"Nonconformity Score ({scores_label})")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            output_dir / f"{scores_label}_scores_distribution_{model_name}.png",
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
        axs[i].set_xlabel(f"Nonconformity Score ({scores_label})")
        axs[i].set_ylabel("Density")
        axs[i].legend()
        axs[i].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / f"{scores_label}_scores_distribution_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# TODO: understand and fix the issue with replicated figures when running multiple times
def plot_CI(
    exp_idx: int,
    mu_hat: float,
    mu_hat_lower: float,
    mu_hat_upper: float,
    mu_true: float,
    ax: plt.Axes,
) -> plt.Axes:
    """Plot a single confidence interval on the given Axes."""
    fontsize = 11
    hat_color = "green" if mu_hat_lower < mu_true < mu_hat_upper else "red"
    line_color = "blue"

    ax.hlines(exp_idx, mu_hat_lower, mu_hat_upper, colors=line_color, linewidth=2)

    ax.plot(
        mu_hat,
        exp_idx,
        "o",
        color=hat_color,
        markersize=12,
        label=f"μ̂ (Exp {exp_idx})",
    )

    ax.plot(mu_true, exp_idx, "^", color=line_color, markersize=12)

    ax.text(
        mu_hat,
        exp_idx + 0.25,
        f"μ̂={mu_hat:.2f}",
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color=hat_color,
    )
    ax.text(
        mu_true,
        exp_idx + 0.25,
        f"μ={mu_true:.2f}",
        ha="center",
        va="top",
        fontsize=fontsize,
        color=line_color,
    )
    ax.text(
        mu_hat_lower,
        exp_idx + 0.25,
        f"\nCI: [{mu_hat_lower:.2f}, {mu_hat_upper:.2f}]",
        ha="left",
        va="bottom",
        fontsize=fontsize,
        color=hat_color,
    )
    return ax


def plot_confidence_intervals(
    mu_hat_values: List[float],
    mu_hat_lower_bounds: List[float],
    mu_hat_upper_bounds: List[float],
    mu_true_list: List[float],
    model_name: str,
    empirical_coverage: float,
    output_dir: Path = Path("plots"),
) -> None:
    """Plot confidence intervals for mu_hat estimates."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _experiment_idxs = range(len(mu_hat_values))

    for plot_id in range(1, 6):
        if len(_experiment_idxs) == 0:
            break
        _experiments_to_plot = np.sort(
            np.random.choice(
                _experiment_idxs, size=min(5, len(_experiment_idxs)), replace=False
            )
        )
        _experiment_idxs = [
            _ for _ in _experiment_idxs if _ not in _experiments_to_plot
        ]

        mu_hat_values_to_plot = np.array(mu_hat_values)[_experiments_to_plot].tolist()
        mu_true_list_to_plot = np.array(mu_true_list)[_experiments_to_plot].tolist()
        mu_hat_lower_bounds_to_plot = np.array(mu_hat_lower_bounds)[
            _experiments_to_plot
        ].tolist()
        mu_hat_upper_bounds_to_plot = np.array(mu_hat_upper_bounds)[
            _experiments_to_plot
        ].tolist()
        mu_hat_lower_bounds_to_plot = np.array(mu_hat_lower_bounds)[
            _experiments_to_plot
        ].tolist()

        fig, ax = plt.subplots(figsize=(12, 6))
        for exp_idx, mu_hat, mu_hat_lower, mu_hat_upper, mu_true in zip(
            _experiments_to_plot,
            mu_hat_values_to_plot,
            mu_hat_lower_bounds_to_plot,
            mu_hat_upper_bounds_to_plot,
            mu_true_list_to_plot,
        ):
            ax = plot_CI(exp_idx, mu_hat, mu_hat_lower, mu_hat_upper, mu_true, ax)

        ax.text(
            0.95,
            1.05,
            f"Empirical coverage: {empirical_coverage*100:.4f}% vs 1-α=68%",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        ax.set_yticks([*range(1, len(mu_hat_values) + 1)] + [len(mu_hat_values) + 0.5])
        ax.set_yticklabels(
            [f"Exp {i}" for i in range(1, len(mu_hat_values) + 1)] + [""]
        )
        ax.set_xlabel("μ value")
        ax.set_ylabel("Experiment")
        ax.set_title(f"Confidence Intervals: {model_name}")
        ax.grid(alpha=0.3, axis="x")
        ax.invert_yaxis()

        plt.tight_layout()
        plt.savefig(
            output_dir / f"test_CI_plots{plot_id}_{model_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


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
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    dx = 0.05 * (x_max - x_min)
    dy = 0.05 * (y_max - y_min)
    x_grid, y_grid = np.mgrid[
        (x_min - dx) : (x_max + dx) : 200j,
        (y_min - dy) : (y_max + dy) : 200j,
    ]
    grid = np.vstack([x_grid.ravel(), y_grid.ravel()])
    background_kde = gaussian_kde(background.T)
    signal_kde = gaussian_kde(signal.T)
    background_density = background_kde(grid).reshape(x_grid.shape)
    signal_density = signal_kde(grid).reshape(x_grid.shape)
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
