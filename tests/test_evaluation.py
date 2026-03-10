"""Unit tests for conformal_predictions.evaluation.

Tests cover:
- performance metrics (metrics.py)
- calibration quality metrics (metrics.py)
- ci_score challenge formula (metrics.py)
- Phase 4: plot functions (plots.py)
- Phase 4: run report generation (reports.py)
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from conformal_predictions.evaluation.metrics import (
    METRIC_REGISTRY,
    compute_calibration_metrics,
    compute_ci_score,
    compute_coverage,
    compute_mean_width,
    compute_per_example_loss,
    compute_performance_metrics,
)

# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------


class TestPerformanceMetrics:
    """Check compute_performance_metrics against sklearn ground truth."""

    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=200)
        y_proba = rng.uniform(0, 1, size=200)
        y_pred = (y_proba > 0.5).astype(int)
        return y_true, y_pred, y_proba

    def test_all_metrics_present(self, data):
        y_true, y_pred, y_proba = data
        result = compute_performance_metrics(y_true, y_pred, y_proba)
        for key in (
            "loss",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "pr_auc",
            "roc_auc",
        ):
            assert key in result
        assert result["loss_name"] == "binary_cross_entropy"

    def test_accuracy_matches_sklearn(self, data):
        y_true, y_pred, y_proba = data
        result = compute_performance_metrics(y_true, y_pred, y_proba)
        expected = accuracy_score(y_true, y_pred)
        assert result["accuracy"] == pytest.approx(expected)

    def test_precision_matches_sklearn(self, data):
        y_true, y_pred, y_proba = data
        result = compute_performance_metrics(y_true, y_pred, y_proba)
        expected = precision_score(y_true, y_pred, zero_division=0)
        assert result["precision"] == pytest.approx(expected)

    def test_recall_matches_sklearn(self, data):
        y_true, y_pred, y_proba = data
        result = compute_performance_metrics(y_true, y_pred, y_proba)
        expected = recall_score(y_true, y_pred, zero_division=0)
        assert result["recall"] == pytest.approx(expected)

    def test_f1_matches_sklearn(self, data):
        y_true, y_pred, y_proba = data
        result = compute_performance_metrics(y_true, y_pred, y_proba)
        expected = f1_score(y_true, y_pred, zero_division=0)
        assert result["f1"] == pytest.approx(expected)

    def test_loss_matches_sklearn(self, data):
        y_true, y_pred, y_proba = data
        result = compute_performance_metrics(y_true, y_pred, y_proba)
        expected = log_loss(y_true, y_proba)
        assert result["loss"] == pytest.approx(expected)

    def test_roc_auc_matches_sklearn(self, data):
        y_true, y_pred, y_proba = data
        result = compute_performance_metrics(y_true, y_pred, y_proba)
        expected = roc_auc_score(y_true, y_proba)
        assert result["roc_auc"] == pytest.approx(expected)

    def test_pr_auc_matches_sklearn(self, data):
        y_true, y_pred, y_proba = data
        result = compute_performance_metrics(y_true, y_pred, y_proba)
        expected = average_precision_score(y_true, y_proba)
        assert result["pr_auc"] == pytest.approx(expected)

    def test_subset_metrics(self, data):
        y_true, y_pred, y_proba = data
        result = compute_performance_metrics(
            y_true, y_pred, y_proba, metric_names=["accuracy", "f1"]
        )
        assert "accuracy" in result
        assert "f1" in result
        assert "roc_auc" not in result

    def test_unknown_metric_raises(self, data):
        y_true, y_pred, y_proba = data
        with pytest.raises(KeyError, match="Unknown metric"):
            compute_performance_metrics(
                y_true, y_pred, y_proba, metric_names=["nonexistent"]
            )

    def test_no_proba(self, data):
        y_true, y_pred, _ = data
        result = compute_performance_metrics(y_true, y_pred, y_proba=None)
        assert math.isnan(result["loss"])
        assert math.isnan(result["roc_auc"])
        assert not math.isnan(result["accuracy"])


# ---------------------------------------------------------------------------
# Calibration quality metrics
# ---------------------------------------------------------------------------


class TestCoverage:
    def test_full_coverage(self):
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([10.0, 10.0, 10.0])
        y_true = np.array([5.0, 5.0, 5.0])
        assert compute_coverage(lower, upper, y_true) == pytest.approx(1.0)

    def test_no_coverage(self):
        lower = np.array([6.0, 6.0])
        upper = np.array([10.0, 10.0])
        y_true = np.array([1.0, 2.0])
        assert compute_coverage(lower, upper, y_true) == pytest.approx(0.0)

    def test_partial_coverage(self):
        lower = np.array([0.0, 5.0])
        upper = np.array([3.0, 10.0])
        y_true = np.array([1.0, 1.0])
        # first: 0 < 1 < 3  ✓, second: 5 < 1 is False  ✗
        assert compute_coverage(lower, upper, y_true) == pytest.approx(0.5)


class TestMeanWidth:
    def test_uniform(self):
        lower = np.array([0.0, 1.0])
        upper = np.array([2.0, 3.0])
        assert compute_mean_width(lower, upper) == pytest.approx(2.0)


class TestCIScore:
    def test_perfect_coverage(self):
        alpha = 0.3173
        c0 = 1 - alpha
        score = compute_ci_score(coverage=c0, width=0.5, alpha=alpha)
        # penalty = 1.0, so s = -log(0.5 + 1e-6)
        expected = -math.log(0.5 + 1e-6)
        assert score == pytest.approx(expected, rel=1e-4)

    def test_penalty_increases_with_coverage_gap(self):
        alpha = 0.3173
        score_good = compute_ci_score(coverage=1 - alpha, width=0.5, alpha=alpha)
        score_bad = compute_ci_score(coverage=0.5, width=0.5, alpha=alpha)
        # Worse coverage → lower score (more negative)
        assert score_bad < score_good

    def test_wider_intervals_lower_score(self):
        alpha = 0.3173
        s_narrow = compute_ci_score(coverage=0.7, width=0.1, alpha=alpha)
        s_wide = compute_ci_score(coverage=0.7, width=1.0, alpha=alpha)
        assert s_wide < s_narrow


class TestCalibrationMetrics:
    def test_combined(self):
        lower = np.array([0.0, 0.0, 0.0, 6.0])
        upper = np.array([5.0, 5.0, 5.0, 10.0])
        y_true = np.array([2.0, 3.0, 4.0, 1.0])

        result = compute_calibration_metrics(lower, upper, y_true, alpha=0.3173)
        assert "coverage" in result
        assert "width" in result
        assert "ci_score" in result
        assert result["coverage"] == pytest.approx(0.75)
        # widths: 5, 5, 5, 4 → mean = 4.75
        assert result["width"] == pytest.approx(4.75)


class TestMetricRegistry:
    def test_all_registered(self):
        expected = {
            "loss",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "pr_auc",
            "roc_auc",
        }
        assert set(METRIC_REGISTRY.keys()) == expected


# ---------------------------------------------------------------------------
# Phase 4: plot functions
# ---------------------------------------------------------------------------

matplotlib.use("Agg")  # non-interactive backend for tests


class TestPlotFunctions:
    """Each plot function must save a PNG and return a matplotlib Figure."""

    @pytest.fixture
    def rng(self):
        return np.random.default_rng(0)

    def test_plot_roc_curve(self, tmp_path, rng):
        from conformal_predictions.evaluation.plots import plot_roc_curve

        y_true = rng.integers(0, 2, size=100)
        y_score = rng.uniform(0, 1, size=100)
        out = tmp_path / "roc.png"
        fig = plot_roc_curve(y_true, y_score, "TestModel", output_path=out)
        assert out.exists(), "PNG file was not created"
        import matplotlib.pyplot as plt

        assert isinstance(fig, plt.Figure)

    def test_plot_pr_curve(self, tmp_path, rng):
        from conformal_predictions.evaluation.plots import plot_pr_curve

        y_true = rng.integers(0, 2, size=100)
        y_score = rng.uniform(0, 1, size=100)
        out = tmp_path / "pr.png"
        fig = plot_pr_curve(y_true, y_score, "TestModel", output_path=out)
        assert out.exists()
        import matplotlib.pyplot as plt

        assert isinstance(fig, plt.Figure)

    def test_plot_nonconformity_scores(self, tmp_path, rng):
        from conformal_predictions.evaluation.plots import plot_nonconformity_scores

        scores = rng.uniform(-1, 1, size=200)
        out = tmp_path / "ncs.png"
        fig = plot_nonconformity_scores(scores, alpha=0.3173, output_path=out)
        assert out.exists()
        import matplotlib.pyplot as plt

        assert isinstance(fig, plt.Figure)

    def test_plot_mu_hat_distribution(self, tmp_path, rng):
        from conformal_predictions.evaluation.plots import plot_mu_hat_distribution

        vals = rng.normal(1.0, 0.2, size=300)
        out = tmp_path / "mu_hat.png"
        fig = plot_mu_hat_distribution(vals, output_path=out, model_name="GLM")
        assert out.exists()
        import matplotlib.pyplot as plt

        assert isinstance(fig, plt.Figure)

    def test_plot_ci_coverage(self, tmp_path):
        from conformal_predictions.evaluation.plots import plot_ci_coverage

        coverages = {"GLM": 0.72, "RF": 0.68, "MLP": 0.65}
        out = tmp_path / "ci_cov.png"
        fig = plot_ci_coverage(coverages, target_coverage=0.6827, output_path=out)
        assert out.exists()
        import matplotlib.pyplot as plt

        assert isinstance(fig, plt.Figure)

    def test_plot_ci_width_distribution(self, tmp_path, rng):
        from conformal_predictions.evaluation.plots import plot_ci_width_distribution

        widths = rng.exponential(0.3, size=200)
        out = tmp_path / "ci_width.png"
        fig = plot_ci_width_distribution(widths, output_path=out, model_name="GLM")
        assert out.exists()
        import matplotlib.pyplot as plt

        assert isinstance(fig, plt.Figure)

    def test_no_output_path_returns_figure(self, rng):
        """When output_path is None, figure is returned and no file is written."""
        import matplotlib.pyplot as plt

        from conformal_predictions.evaluation.plots import plot_roc_curve

        y_true = rng.integers(0, 2, size=50)
        y_score = rng.uniform(0, 1, size=50)
        fig = plot_roc_curve(y_true, y_score, "M")
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_ax_parameter_respected(self, rng):
        """Passing ax= should draw on that axes and return the correct Figure."""
        import matplotlib.pyplot as plt

        from conformal_predictions.evaluation.plots import plot_roc_curve

        y_true = rng.integers(0, 2, size=50)
        y_score = rng.uniform(0, 1, size=50)
        fig_ext, ax_ext = plt.subplots()
        fig_returned = plot_roc_curve(y_true, y_score, "M", ax=ax_ext)
        assert fig_returned is fig_ext
        plt.close("all")


# ---------------------------------------------------------------------------
# Phase 4: run report generation
# ---------------------------------------------------------------------------


class TestRunReport:
    """generate_run_report must write a valid Markdown file with key sections."""

    def _make_ctx(self, tmp_path: Path):
        """Build a minimal RunContext without needing a full config."""
        from conformal_predictions.mlops.run_context import RunContext

        ctx = RunContext(
            run_id="abc12345",
            timestamp="2026-03-10T12:00:00Z",
            config_snapshot={},
            config_path="configs/train_toy.yaml",
            dataset="toy",
            git_commit="deadbeef",
            output_dir=tmp_path,
        )
        # Register a fake plot artifact
        ctx.save_artifact(
            "plots/GLM_roc_curve.png",
            type="plot",
            format="png",
            description="ROC curve — GLM",
        )
        return ctx

    def _make_metrics(self):
        return {
            "GLM": {
                "performance": {
                    "accuracy": 0.95,
                    "precision": 0.93,
                    "recall": 0.91,
                    "f1": 0.92,
                    "roc_auc": 0.97,
                    "pr_auc": 0.96,
                },
                "calibration": {
                    "coverage": 0.68,
                    "width": 0.21,
                    "ci_score": 3.14,
                },
            }
        }

    def test_report_file_created(self, tmp_path):
        from conformal_predictions.evaluation.reports import generate_run_report

        ctx = self._make_ctx(tmp_path)
        out = tmp_path / "report.md"
        result = generate_run_report(ctx, metrics=self._make_metrics(), output_path=out)
        assert result == out
        assert out.exists()

    def test_report_contains_run_id(self, tmp_path):
        from conformal_predictions.evaluation.reports import generate_run_report

        ctx = self._make_ctx(tmp_path)
        out = tmp_path / "report.md"
        generate_run_report(ctx, metrics=self._make_metrics(), output_path=out)
        content = out.read_text()
        assert "abc12345" in content

    def test_report_contains_metadata_fields(self, tmp_path):
        from conformal_predictions.evaluation.reports import generate_run_report

        ctx = self._make_ctx(tmp_path)
        out = tmp_path / "report.md"
        generate_run_report(ctx, metrics=self._make_metrics(), output_path=out)
        content = out.read_text()
        for field in ("Run Metadata", "Dataset", "Git Commit", "2026-03-10"):
            assert field in content, f"Expected '{field}' in report"

    def test_report_contains_metrics_table(self, tmp_path):
        from conformal_predictions.evaluation.reports import generate_run_report

        ctx = self._make_ctx(tmp_path)
        out = tmp_path / "report.md"
        generate_run_report(ctx, metrics=self._make_metrics(), output_path=out)
        content = out.read_text()
        assert "Classification Metrics" in content
        assert "GLM" in content
        assert "0.9500" in content  # accuracy

    def test_report_contains_calibration_table(self, tmp_path):
        from conformal_predictions.evaluation.reports import generate_run_report

        ctx = self._make_ctx(tmp_path)
        out = tmp_path / "report.md"
        generate_run_report(ctx, metrics=self._make_metrics(), output_path=out)
        content = out.read_text()
        assert "Calibration Quality" in content
        assert "Coverage" in content

    def test_report_contains_plot_references(self, tmp_path):
        from conformal_predictions.evaluation.reports import generate_run_report

        ctx = self._make_ctx(tmp_path)
        out = tmp_path / "report.md"
        generate_run_report(ctx, metrics=self._make_metrics(), output_path=out)
        content = out.read_text()
        assert "GLM_roc_curve.png" in content

    def test_partial_report_no_metrics(self, tmp_path):
        """Report with no metrics still produces a valid file with metadata."""
        from conformal_predictions.evaluation.reports import generate_run_report

        ctx = self._make_ctx(tmp_path)
        out = tmp_path / "report.md"
        generate_run_report(ctx, metrics=None, output_path=out)
        content = out.read_text()
        assert "abc12345" in content
        assert "Classification Metrics" not in content


# ---------------------------------------------------------------------------
# Phase 4.5: new plot functions
# ---------------------------------------------------------------------------


class TestNewPlotFunctions:
    """Tests for Phase 4.5 new plot functions."""

    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42)

    def test_plot_target_distribution(self, tmp_path, rng):
        import matplotlib.pyplot as plt

        from conformal_predictions.evaluation.plots import plot_target_distribution

        y = rng.integers(0, 2, size=200)
        out = tmp_path / "target_dist.png"
        fig = plot_target_distribution(y, output_path=out)
        assert out.exists()
        assert isinstance(fig, plt.Figure)

    def test_plot_target_distribution_log_scale(self, tmp_path):
        """Imbalanced data should trigger log scale."""
        import matplotlib.pyplot as plt

        from conformal_predictions.evaluation.plots import plot_target_distribution

        y = np.concatenate([np.zeros(1000), np.ones(10)])
        out = tmp_path / "target_dist_imb.png"
        fig = plot_target_distribution(y, output_path=out)
        assert out.exists()
        assert isinstance(fig, plt.Figure)

    def test_plot_predictions_ecdf(self, tmp_path, rng):
        import matplotlib.pyplot as plt

        from conformal_predictions.evaluation.plots import plot_predictions_ecdf

        proba_train = rng.uniform(0, 1, size=200)
        proba_val = rng.uniform(0, 1, size=100)
        out = tmp_path / "ecdf.png"
        fig = plot_predictions_ecdf(proba_train, proba_val, output_path=out)
        assert out.exists()
        assert isinstance(fig, plt.Figure)

    def test_plot_nonconformity_ecdf(self, tmp_path, rng):
        import matplotlib.pyplot as plt

        from conformal_predictions.evaluation.plots import plot_nonconformity_ecdf

        scores = rng.normal(0, 1, size=200)
        out = tmp_path / "nc_ecdf.png"
        fig = plot_nonconformity_ecdf(scores, output_path=out)
        assert out.exists()
        assert isinstance(fig, plt.Figure)

    def test_plot_nonconformity_by_class(self, tmp_path, rng):
        import matplotlib.pyplot as plt

        from conformal_predictions.evaluation.plots import plot_nonconformity_by_class

        s0 = rng.normal(0, 1, size=100)
        s1 = rng.normal(1, 1, size=80)
        out = tmp_path / "nc_by_class.png"
        fig = plot_nonconformity_by_class(s0, s1, output_path=out)
        assert out.exists()
        assert isinstance(fig, plt.Figure)

    def test_plot_distribution(self, tmp_path, rng):
        import matplotlib.pyplot as plt

        from conformal_predictions.evaluation.plots import plot_distribution

        vals = rng.exponential(1, size=300)
        out = tmp_path / "dist.png"
        fig = plot_distribution(vals, output_path=out, title="Test Dist")
        assert out.exists()
        assert isinstance(fig, plt.Figure)

    def test_plot_distribution_empty(self, tmp_path):
        import matplotlib.pyplot as plt

        from conformal_predictions.evaluation.plots import plot_distribution

        out = tmp_path / "empty.png"
        fig = plot_distribution([], output_path=out, title="Empty")
        assert out.exists()
        assert isinstance(fig, plt.Figure)

    def test_plot_confusion_matrix(self, tmp_path, rng):
        import matplotlib.pyplot as plt

        from conformal_predictions.evaluation.plots import plot_confusion_matrix

        y_true = rng.integers(0, 2, size=100)
        y_pred = rng.integers(0, 2, size=100)
        out = tmp_path / "cm.png"
        fig = plot_confusion_matrix(y_true, y_pred, output_path=out)
        assert out.exists()
        assert isinstance(fig, plt.Figure)

    def test_roc_curve_with_train(self, tmp_path, rng):
        """ROC curve with optional train overlay."""
        import matplotlib.pyplot as plt

        from conformal_predictions.evaluation.plots import plot_roc_curve

        y_true = rng.integers(0, 2, size=100)
        y_score = rng.uniform(0, 1, size=100)
        y_true_tr = rng.integers(0, 2, size=200)
        y_score_tr = rng.uniform(0, 1, size=200)
        out = tmp_path / "roc_joint.png"
        fig = plot_roc_curve(
            y_true,
            y_score,
            "M",
            output_path=out,
            y_true_train=y_true_tr,
            y_score_train=y_score_tr,
        )
        assert out.exists()
        assert isinstance(fig, plt.Figure)

    def test_pr_curve_with_train(self, tmp_path, rng):
        """PR curve with optional train overlay."""
        import matplotlib.pyplot as plt

        from conformal_predictions.evaluation.plots import plot_pr_curve

        y_true = rng.integers(0, 2, size=100)
        y_score = rng.uniform(0, 1, size=100)
        y_true_tr = rng.integers(0, 2, size=200)
        y_score_tr = rng.uniform(0, 1, size=200)
        out = tmp_path / "pr_joint.png"
        fig = plot_pr_curve(
            y_true,
            y_score,
            "M",
            output_path=out,
            y_true_train=y_true_tr,
            y_score_train=y_score_tr,
        )
        assert out.exists()
        assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# Phase 4.5: per-example loss
# ---------------------------------------------------------------------------


class TestPerExampleLoss:
    def test_basic_shape(self):
        y_true = np.array([0, 1, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.2, 0.8])
        losses = compute_per_example_loss(y_true, y_proba)
        assert losses.shape == (4,)
        assert np.all(losses >= 0)

    def test_perfect_predictions(self):
        y_true = np.array([0, 1, 0, 1])
        y_proba = np.array([1e-6, 1.0 - 1e-6, 1e-6, 1.0 - 1e-6])
        losses = compute_per_example_loss(y_true, y_proba)
        assert np.all(losses < 0.01)

    def test_worst_case(self):
        y_true = np.array([0, 1])
        y_proba = np.array([0.999, 0.001])
        losses = compute_per_example_loss(y_true, y_proba)
        assert np.all(losses > 5.0)


# ---------------------------------------------------------------------------
# Phase 4.5: error analysis
# ---------------------------------------------------------------------------


class TestErrorAnalysis:
    def test_build_top_errors_table(self):
        from conformal_predictions.evaluation.error_analysis import (
            build_top_errors_table,
        )

        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=200)
        y_proba = rng.uniform(0, 1, size=200)
        y_pred = (y_proba > 0.5).astype(int)
        df = build_top_errors_table(y_true, y_pred, y_proba, N=50)
        assert len(df) == 50
        assert "per_example_loss" in df.columns
        assert "true_label" in df.columns
        assert "confidence" in df.columns
        # Should be sorted by loss descending
        assert df["per_example_loss"].iloc[0] >= df["per_example_loss"].iloc[-1]

    def test_build_top_errors_with_features(self):
        from conformal_predictions.evaluation.error_analysis import (
            build_top_errors_table,
        )

        rng = np.random.default_rng(42)
        n = 100
        y_true = rng.integers(0, 2, size=n)
        y_proba = rng.uniform(0, 1, size=n)
        y_pred = (y_proba > 0.5).astype(int)
        X = rng.standard_normal((n, 3))
        df = build_top_errors_table(
            y_true, y_pred, y_proba, X=X, feature_names=["a", "b", "c"], N=10
        )
        assert len(df) == 10
        assert "a" in df.columns
        assert "b" in df.columns
        assert "c" in df.columns

    def test_build_top_errors_n_larger_than_data(self):
        from conformal_predictions.evaluation.error_analysis import (
            build_top_errors_table,
        )

        y_true = np.array([0, 1])
        y_proba = np.array([0.9, 0.1])
        y_pred = np.array([1, 0])
        df = build_top_errors_table(y_true, y_pred, y_proba, N=100)
        assert len(df) == 2
