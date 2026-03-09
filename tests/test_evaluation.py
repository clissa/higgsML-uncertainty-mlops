"""Unit tests for conformal_predictions.evaluation.

Tests cover:
- performance metrics (metrics.py)
- calibration quality metrics (metrics.py)
- ci_score challenge formula (metrics.py)
"""

from __future__ import annotations

import math

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
