"""Per-run plot functions for the conformal-prediction evaluation pipeline.

All functions are pure: they accept numpy arrays, optionally save to
``output_path``, and return a ``matplotlib.Figure`` object.  Callers are
responsible for registering saved files via ``RunContext.save_artifact()``.

Usage::

    from conformal_predictions.evaluation.plots import plot_roc_curve
    fig = plot_roc_curve(y_true, y_score, "GLM",
                         output_path=plots_dir / "GLM_roc_curve.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

mpl.rcParams["font.size"] = 13

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PathLike = Union[str, Path, None]


def _save_and_close(fig: plt.Figure, output_path: PathLike, dpi: int = 150) -> None:
    """Save *fig* to *output_path* and close it (no-op when path is None)."""
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    model_name: str,
    output_path: PathLike = None,
    ax: Optional[plt.Axes] = None,
    dpi: int = 150,
    y_true_train: Optional[np.ndarray] = None,
    y_score_train: Optional[np.ndarray] = None,
) -> plt.Figure:
    """ROC curve for *model_name*.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        True binary labels (validation/test set).
    y_score : array-like of shape (n,)
        Predicted probabilities for the positive class (val/test).
    model_name : str
        Used in the plot title.
    output_path : str or Path, optional
        If provided, the figure is saved here as PNG.
    ax : matplotlib.Axes, optional
        Existing axes to draw on; a new figure is created when *None*.
    dpi : int
        Figure DPI when saving.
    y_true_train : array-like, optional
        Train-set true labels. When provided, both train and val curves
        are drawn on the same axes.
    y_score_train : array-like, optional
        Train-set predicted probabilities.

    Returns
    -------
    matplotlib.Figure
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    # Optional train curve
    if y_true_train is not None and y_score_train is not None:
        fpr_tr, tpr_tr, _ = roc_curve(y_true_train, y_score_train)
        roc_auc_tr = auc(fpr_tr, tpr_tr)
        ax.plot(
            fpr_tr,
            tpr_tr,
            lw=1.5,
            alpha=0.6,
            label=f"Train AUC = {roc_auc_tr:.3f}",
            linestyle="--",
        )

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    label_prefix = "Val " if y_true_train is not None else ""
    ax.plot(fpr, tpr, lw=2, label=f"{label_prefix}AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    if own_fig:
        fig.tight_layout()
        _save_and_close(fig, output_path, dpi=dpi)

    return fig


def plot_pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    model_name: str,
    output_path: PathLike = None,
    ax: Optional[plt.Axes] = None,
    dpi: int = 150,
    y_true_train: Optional[np.ndarray] = None,
    y_score_train: Optional[np.ndarray] = None,
) -> plt.Figure:
    """Precision-Recall curve for *model_name*.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        True binary labels (val/test).
    y_score : array-like of shape (n,)
        Predicted probabilities for the positive class (val/test).
    model_name : str
        Used in the plot title.
    output_path : str or Path, optional
        If provided, the figure is saved as PNG.
    ax : matplotlib.Axes, optional
        Existing axes; created when *None*.
    dpi : int
        Figure DPI when saving.
    y_true_train : array-like, optional
        Train-set true labels. When provided, both curves are drawn.
    y_score_train : array-like, optional
        Train-set predicted probabilities.

    Returns
    -------
    matplotlib.Figure
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    # Optional train curve
    if y_true_train is not None and y_score_train is not None:
        prec_tr, rec_tr, _ = precision_recall_curve(y_true_train, y_score_train)
        pr_auc_tr = auc(rec_tr, prec_tr)
        ax.plot(
            rec_tr,
            prec_tr,
            lw=1.5,
            alpha=0.6,
            label=f"Train AUC = {pr_auc_tr:.3f}",
            linestyle="--",
        )

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)

    label_prefix = "Val " if y_true_train is not None else ""
    ax.plot(recall, precision, lw=2, label=f"{label_prefix}AUC = {pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR Curve — {model_name}")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    if own_fig:
        fig.tight_layout()
        _save_and_close(fig, output_path, dpi=dpi)

    return fig


def plot_nonconformity_scores(
    scores: Sequence[float],
    alpha: float,
    output_path: PathLike = None,
    ax: Optional[plt.Axes] = None,
    model_name: str = "",
    dpi: int = 150,
) -> plt.Figure:
    """Histogram of nonconformity scores with the coverage quantile marked.

    Parameters
    ----------
    scores : sequence of float
        Nonconformity scores from the calibration set.
    alpha : float
        Significance level; quantile at ``1 - alpha`` is drawn.
    output_path : str or Path, optional
        If provided, the figure is saved as PNG.
    ax : matplotlib.Axes, optional
        Existing axes; created when *None*.
    model_name : str
        Used in the plot title suffix.
    dpi : int
        Figure DPI when saving.

    Returns
    -------
    matplotlib.Figure
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    scores_arr = np.asarray(scores, dtype=float)
    q = float(np.quantile(scores_arr, 1.0 - alpha))

    ax.hist(scores_arr, bins=30, alpha=0.7, color="steelblue", density=True)
    if len(scores_arr) > 1:
        kde = gaussian_kde(scores_arr)
        xs = np.linspace(scores_arr.min(), scores_arr.max(), 300)
        ax.plot(xs, kde(xs), "k-", lw=1.5, label="KDE")

    ax.axvline(
        q,
        color="crimson",
        lw=2,
        linestyle="--",
        label=f"q({1 - alpha:.3f}) = {q:.4f}",
    )
    title = "Nonconformity Scores"
    if model_name:
        title += f" — {model_name}"
    ax.set_title(title)
    ax.set_xlabel("Nonconformity Score")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)

    if own_fig:
        fig.tight_layout()
        _save_and_close(fig, output_path, dpi=dpi)

    return fig


def plot_mu_hat_distribution(
    mu_hat_values: Sequence[float],
    output_path: PathLike = None,
    ax: Optional[plt.Axes] = None,
    model_name: str = "",
    dpi: int = 150,
) -> plt.Figure:
    """Distribution of estimated signal strength μ̂ across pseudo-experiments.

    Parameters
    ----------
    mu_hat_values : sequence of float
        One μ̂ estimate per test pseudo-experiment.
    output_path : str or Path, optional
        If provided, the figure is saved as PNG.
    ax : matplotlib.Axes, optional
        Existing axes; created when *None*.
    model_name : str
        Used in the plot title.
    dpi : int
        Figure DPI when saving.

    Returns
    -------
    matplotlib.Figure
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    vals = np.asarray(mu_hat_values, dtype=float)
    if len(vals) == 0:
        ax.set_title(f"μ̂ Distribution — {model_name} (no data)")
        if own_fig:
            fig.tight_layout()
            _save_and_close(fig, output_path, dpi=dpi)
        return fig

    median = float(np.median(vals))
    mean = float(np.mean(vals))
    q16 = float(np.quantile(vals, 0.16))
    q84 = float(np.quantile(vals, 0.84))

    ax.hist(vals, bins=30, alpha=0.7, color="steelblue", density=True)
    if len(vals) > 1:
        kde = gaussian_kde(vals)
        xs = np.linspace(vals.min(), vals.max(), 300)
        ax.plot(xs, kde(xs), "k-", lw=1.5, label="KDE")

    ax.axvline(
        median, color="orange", lw=2, linestyle="--", label=f"median={median:.3f}"
    )
    ax.axvline(mean, color="red", lw=2, linestyle=":", label=f"mean={mean:.3f}")
    ax.axvline(q16, color="green", lw=1.5, linestyle="--", label=f"q16={q16:.3f}")
    ax.axvline(q84, color="purple", lw=1.5, linestyle="--", label=f"q84={q84:.3f}")

    title = r"$\hat{\mu}$ Distribution"
    if model_name:
        title += f" — {model_name}"
    ax.set_title(title)
    ax.set_xlabel(r"$\hat{\mu}$")
    ax.set_ylabel("Density")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    if own_fig:
        fig.tight_layout()
        _save_and_close(fig, output_path, dpi=dpi)

    return fig


def plot_ci_coverage(
    coverages: Dict[str, float],
    target_coverage: float,
    output_path: PathLike = None,
    ax: Optional[plt.Axes] = None,
    dpi: int = 150,
) -> plt.Figure:
    """Bar chart of empirical coverage per model vs the target coverage.

    Parameters
    ----------
    coverages : dict
        ``{model_name: empirical_coverage}`` mapping.
    target_coverage : float
        Nominal target, i.e. ``1 - alpha``.
    output_path : str or Path, optional
        If provided, the figure is saved as PNG.
    ax : matplotlib.Axes, optional
        Existing axes; created when *None*.
    dpi : int
        Figure DPI when saving.

    Returns
    -------
    matplotlib.Figure
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(max(5, len(coverages) * 1.5), 4))
    else:
        fig = ax.get_figure()

    names = list(coverages.keys())
    vals = [coverages[n] for n in names]
    colors = ["#59A14F" if v >= target_coverage else "#E15759" for v in vals]

    bars = ax.bar(names, vals, color=colors, alpha=0.8, edgecolor="white")
    ax.axhline(
        target_coverage,
        color="navy",
        lw=2,
        linestyle="--",
        label=f"Target = {target_coverage:.3f}",
    )
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.005,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax.set_ylim(0, min(1.05, max(vals) + 0.1))
    ax.set_ylabel("Empirical Coverage")
    ax.set_title("CI Coverage by Model")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    if own_fig:
        fig.tight_layout()
        _save_and_close(fig, output_path, dpi=dpi)

    return fig


def plot_ci_width_distribution(
    widths: Sequence[float],
    output_path: PathLike = None,
    ax: Optional[plt.Axes] = None,
    model_name: str = "",
    dpi: int = 150,
) -> plt.Figure:
    """Histogram of CI widths across test pseudo-experiments.

    Parameters
    ----------
    widths : sequence of float
        ``upper_i - lower_i`` per pseudo-experiment.
    output_path : str or Path, optional
        If provided, the figure is saved as PNG.
    ax : matplotlib.Axes, optional
        Existing axes; created when *None*.
    model_name : str
        Used in the plot title.
    dpi : int
        Figure DPI when saving.

    Returns
    -------
    matplotlib.Figure
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    w = np.asarray(widths, dtype=float)
    mean_w = float(np.mean(w)) if len(w) else float("nan")

    ax.hist(w, bins=30, alpha=0.7, color="mediumpurple", density=True)
    if len(w) > 1:
        ax.axvline(
            mean_w, color="crimson", lw=2, linestyle="--", label=f"mean={mean_w:.4f}"
        )

    title = "CI Width Distribution"
    if model_name:
        title += f" — {model_name}"
    ax.set_title(title)
    ax.set_xlabel("CI Width (upper − lower)")
    ax.set_ylabel("Density")
    if len(w) > 1:
        ax.legend()
    ax.grid(alpha=0.3)

    if own_fig:
        fig.tight_layout()
        _save_and_close(fig, output_path, dpi=dpi)

    return fig


# ---------------------------------------------------------------------------
# Phase 4.5 — new plot functions
# ---------------------------------------------------------------------------


def plot_target_distribution(
    y: np.ndarray,
    output_path: PathLike = None,
    ax: Optional[plt.Axes] = None,
    dpi: int = 150,
    title: str = "Target Distribution",
) -> plt.Figure:
    """Bar chart of class counts with percentage annotations.

    Parameters
    ----------
    y : array-like of shape (n,)
        Binary target labels (0/1).
    output_path : str or Path, optional
        If provided, the figure is saved as PNG.
    ax : matplotlib.Axes, optional
        Existing axes; created when *None*.
    dpi : int
        Figure DPI when saving.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.Figure
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.get_figure()

    y_arr = np.asarray(y, dtype=int)
    n_total = len(y_arr)
    classes, counts = np.unique(y_arr, return_counts=True)

    colors = ["#4E79A7", "#E15759"]
    bars = ax.bar(
        [str(c) for c in classes],
        counts,
        color=colors[: len(classes)],
        alpha=0.85,
        edgecolor="white",
    )
    for bar, cnt in zip(bars, counts):
        pct = 100 * cnt / n_total if n_total > 0 else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            cnt + 0.01 * max(counts),
            f"{cnt}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")

    # Use log scale when imbalance ratio > 10:1
    if len(counts) >= 2 and max(counts) / max(min(counts), 1) > 10:
        ax.set_yscale("log")

    if own_fig:
        fig.tight_layout()
        _save_and_close(fig, output_path, dpi=dpi)

    return fig


def plot_predictions_ecdf(
    y_proba_train: np.ndarray,
    y_proba_val: np.ndarray,
    output_path: PathLike = None,
    ax: Optional[plt.Axes] = None,
    model_name: str = "",
    dpi: int = 150,
) -> plt.Figure:
    """Empirical CDF of prediction scores for train and val overlaid.

    Parameters
    ----------
    y_proba_train : array-like
        Predicted probabilities on training set.
    y_proba_val : array-like
        Predicted probabilities on validation set.
    output_path : str or Path, optional
        If provided, the figure is saved as PNG.
    ax : matplotlib.Axes, optional
        Existing axes; created when *None*.
    model_name : str
        Used in the plot title.
    dpi : int
        Figure DPI when saving.

    Returns
    -------
    matplotlib.Figure
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    for label, proba, color, ls in [
        ("Train", y_proba_train, "#4E79A7", "-"),
        ("Val", y_proba_val, "#E15759", "--"),
    ]:
        arr = np.sort(np.asarray(proba, dtype=float))
        ecdf = np.arange(1, len(arr) + 1) / len(arr)
        ax.plot(arr, ecdf, lw=2, color=color, linestyle=ls, label=label)

    title = "Predictions ECDF"
    if model_name:
        title += f" — {model_name}"
    ax.set_title(title)
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("ECDF")
    ax.legend()
    ax.grid(alpha=0.3)

    if own_fig:
        fig.tight_layout()
        _save_and_close(fig, output_path, dpi=dpi)

    return fig


def plot_nonconformity_ecdf(
    scores: Sequence[float],
    output_path: PathLike = None,
    ax: Optional[plt.Axes] = None,
    model_name: str = "",
    dpi: int = 150,
) -> plt.Figure:
    """ECDF of nonconformity scores.

    Parameters
    ----------
    scores : sequence of float
        Nonconformity scores from the calibration set.
    output_path : str or Path, optional
        If provided, the figure is saved as PNG.
    ax : matplotlib.Axes, optional
        Existing axes; created when *None*.
    model_name : str
        Used in the plot title suffix.
    dpi : int
        Figure DPI when saving.

    Returns
    -------
    matplotlib.Figure
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    arr = np.sort(np.asarray(scores, dtype=float))
    ecdf = np.arange(1, len(arr) + 1) / len(arr)
    ax.plot(arr, ecdf, lw=2, color="steelblue")

    title = "Nonconformity Scores ECDF"
    if model_name:
        title += f" — {model_name}"
    ax.set_title(title)
    ax.set_xlabel("Nonconformity Score")
    ax.set_ylabel("ECDF")
    ax.grid(alpha=0.3)

    if own_fig:
        fig.tight_layout()
        _save_and_close(fig, output_path, dpi=dpi)

    return fig


def plot_nonconformity_by_class(
    scores_class0: Sequence[float],
    scores_class1: Sequence[float],
    output_path: PathLike = None,
    ax: Optional[plt.Axes] = None,
    model_name: str = "",
    dpi: int = 150,
) -> plt.Figure:
    """Nonconformity score distribution split by class label.

    Parameters
    ----------
    scores_class0 : sequence of float
        Scores for class-0 samples.
    scores_class1 : sequence of float
        Scores for class-1 samples.
    output_path : str or Path, optional
        If provided, the figure is saved as PNG.
    ax : matplotlib.Axes, optional
        Existing axes; created when *None*.
    model_name : str
        Used in the plot title suffix.
    dpi : int
        Figure DPI when saving.

    Returns
    -------
    matplotlib.Figure
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    for label, data, color in [
        ("Class 0", scores_class0, "#4E79A7"),
        ("Class 1", scores_class1, "#E15759"),
    ]:
        arr = np.asarray(data, dtype=float)
        if len(arr) > 0:
            ax.hist(arr, bins=30, alpha=0.55, color=color, label=label, density=True)

    title = "Nonconformity Scores by Class"
    if model_name:
        title += f" — {model_name}"
    ax.set_title(title)
    ax.set_xlabel("Nonconformity Score")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)

    if own_fig:
        fig.tight_layout()
        _save_and_close(fig, output_path, dpi=dpi)

    return fig


def plot_distribution(
    values: Sequence[float],
    output_path: PathLike = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Distribution",
    xlabel: str = "Value",
    color: str = "steelblue",
    dpi: int = 150,
) -> plt.Figure:
    """Generic histogram with mean line — reusable for q_low, q_high, widths.

    Parameters
    ----------
    values : sequence of float
        Values to plot.
    output_path : str or Path, optional
        If provided, the figure is saved as PNG.
    ax : matplotlib.Axes, optional
        Existing axes; created when *None*.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    color : str
        Histogram bar colour.
    dpi : int
        Figure DPI when saving.

    Returns
    -------
    matplotlib.Figure
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        ax.set_title(f"{title} (no data)")
        if own_fig:
            fig.tight_layout()
            _save_and_close(fig, output_path, dpi=dpi)
        return fig

    mean_val = float(np.mean(arr))
    ax.hist(arr, bins=30, alpha=0.7, color=color, density=True)
    ax.axvline(
        mean_val, color="crimson", lw=2, linestyle="--", label=f"mean={mean_val:.4f}"
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)

    if own_fig:
        fig.tight_layout()
        _save_and_close(fig, output_path, dpi=dpi)

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: PathLike = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Confusion Matrix",
    dpi: int = 150,
) -> plt.Figure:
    """Confusion matrix heatmap.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        True labels.
    y_pred : array-like of shape (n,)
        Predicted labels.
    output_path : str or Path, optional
        If provided, the figure is saved as PNG.
    ax : matplotlib.Axes, optional
        Existing axes; created when *None*.
    title : str
        Plot title.
    dpi : int
        Figure DPI when saving.

    Returns
    -------
    matplotlib.Figure
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.get_figure()

    cm = confusion_matrix(y_true, y_pred)
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    classes = sorted(
        set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist())
    )
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    if own_fig:
        fig.tight_layout()
        _save_and_close(fig, output_path, dpi=dpi)

    return fig
