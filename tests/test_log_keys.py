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
