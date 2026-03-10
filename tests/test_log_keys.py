"""Unit tests for conformal_predictions.mlops.log_keys."""

from __future__ import annotations

import re

import pytest

from conformal_predictions.mlops.log_keys import (
    _ALLOWED_SECTIONS,
    CALIBRATION,
    EDA,
    ERROR_ANALYSIS,
    EVALUATION,
    PLOTS,
    calib_key,
    plots_key,
    wandb_key,
)


class TestWandbKey:
    """Validate the naming helper."""

    def test_basic_construction(self):
        key = wandb_key(EDA, "plots", "target_distribution")
        assert key == "EDA/plots/target_distribution"

    def test_evaluation_key(self):
        key = wandb_key(EVALUATION, "val", "f1")
        assert key == "Evaluation/val/f1"

    def test_calibration_key(self):
        key = wandb_key(CALIBRATION, "nonconformity", "q_low")
        assert key == "Calibration/nonconformity/q_low"

    def test_error_analysis_key(self):
        key = wandb_key(ERROR_ANALYSIS, "train", "confusion_matrix")
        assert key == "ErrorAnalysis/train/confusion_matrix"

    def test_format_matches_taxonomy(self):
        """All keys must match Section/subsection/name pattern."""
        pattern = re.compile(
            r"^(EDA|Evaluation|Calibration|ErrorAnalysis|Plots)/[^/]+/[^/]+$"
        )
        for section in _ALLOWED_SECTIONS:
            k = wandb_key(section, "test_sub", "test_name")
            assert pattern.match(k), f"Key {k!r} doesn't match taxonomy"

    def test_invalid_section_raises(self):
        with pytest.raises(AssertionError, match="Invalid section"):
            wandb_key("BadSection", "sub", "name")

    def test_all_five_sections_defined(self):
        assert EDA == "EDA"
        assert EVALUATION == "Evaluation"
        assert CALIBRATION == "Calibration"
        assert ERROR_ANALYSIS == "ErrorAnalysis"
        assert PLOTS == "Plots"
        assert len(_ALLOWED_SECTIONS) == 5


class TestCalibKey:
    """Validate the flat Calibration key helper."""

    def test_basic_construction(self):
        key = calib_key("q_low")
        assert key == "Calibration/q_low"

    def test_format_is_two_level(self):
        """calib_key must produce exactly two path components."""
        for name in ("score", "q_high", "ci_width", "coverage", "ci_score"):
            key = calib_key(name)
            parts = key.split("/")
            assert len(parts) == 2, f"Expected 2 levels, got {key!r}"
            assert parts[0] == "Calibration"
            assert parts[1] == name

    def test_no_subsection(self):
        """calib_key must NOT include a subsection (3rd component)."""
        key = calib_key("accuracy")
        assert key.count("/") == 1

    def test_test_ci_width_naming(self):
        """Ensure the test-set CI width key is distinct from per-block ci_width."""
        per_block = calib_key("ci_width")
        test_set = calib_key("test_ci_width")
        assert per_block != test_set


class TestPlotsKey:
    """Validate the flat Plots key helper."""

    def test_basic_construction(self):
        key = plots_key("test_roc_curve")
        assert key == "Plots/test_roc_curve"

    def test_format_is_two_level(self):
        """plots_key must produce exactly two path components."""
        for name in (
            "test_roc_curve",
            "test_pr_curve",
            "predictions_ecdf",
            "mu_hat_distribution_test",
        ):
            key = plots_key(name)
            parts = key.split("/")
            assert len(parts) == 2, f"Expected 2 levels, got {key!r}"
            assert parts[0] == "Plots"
            assert parts[1] == name

    def test_no_subsection(self):
        """plots_key must NOT include a subsection (3rd component)."""
        key = plots_key("predictions_ecdf")
        assert key.count("/") == 1

    def test_test_prefix_for_roc_pr(self):
        """ROC and PR curve keys must carry the test_ prefix."""
        assert plots_key("test_roc_curve").endswith("test_roc_curve")
        assert plots_key("test_pr_curve").endswith("test_pr_curve")
