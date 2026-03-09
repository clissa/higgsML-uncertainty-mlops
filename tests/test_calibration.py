"""Unit tests for conformal_predictions.calibration.

Tests cover:
- nonconformity score computation (scores.py)
- quantile extraction and CI construction (intervals.py)
- CalibrationResult dataclass (strategies.py)
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from conformal_predictions.calibration.intervals import (
    compute_confidence_interval,
    compute_confidence_intervals_from_file,
    extract_quantiles,
    save_intervals,
    save_quantiles,
)
from conformal_predictions.calibration.scores import (
    nonconformity_score,
    save_scores,
)
from conformal_predictions.calibration.strategies import CalibrationResult
from conformal_predictions.config import CalibrationConfig

# ---------------------------------------------------------------------------
# scores.py
# ---------------------------------------------------------------------------


class TestNonconformityScore:
    def test_diff(self):
        # target - pred = 5.0 - 3.0 = 2.0 (plus tiny noise)
        s = nonconformity_score(3.0, 5.0, how="diff")
        assert abs(s - 2.0) < 0.01

    def test_abs(self):
        s = nonconformity_score(5.0, 3.0, how="abs")
        assert abs(s - 2.0) < 0.01

    def test_abs_nonneg(self):
        s = nonconformity_score(3.0, 5.0, how="abs")
        assert s > 0

    def test_invalid_how(self):
        with pytest.raises(ValueError, match="Unknown score method"):
            nonconformity_score(1.0, 2.0, how="bogus")


class TestSaveScores:
    def test_saves_npz_and_csv(self, tmp_path):
        scores = {"ModelA": [0.1, 0.2, 0.3, 0.4, 0.5]}
        npz_path = save_scores(scores, tmp_path)
        assert npz_path.exists()
        data = np.load(npz_path)
        np.testing.assert_allclose(data["ModelA"], [0.1, 0.2, 0.3, 0.4, 0.5])

        csv_path = tmp_path / "nonconformity_scores_distribution.csv"
        assert csv_path.exists()

    def test_no_csv(self, tmp_path):
        scores = {"M": [1.0]}
        save_scores(scores, tmp_path, save_distribution_csv=False)
        csv_files = list(tmp_path.glob("*.csv"))
        assert len(csv_files) == 0


# ---------------------------------------------------------------------------
# intervals.py
# ---------------------------------------------------------------------------


class TestExtractQuantiles:
    def test_asymmetric_diff(self):
        rng = np.random.default_rng(42)
        scores = rng.normal(0, 1, size=10000)
        alpha = 0.3173  # ~1-sigma
        q_low, q_high = extract_quantiles(
            scores, alpha, how="diff", ci_type="asymmetric"
        )
        # For N(0,1), alpha/2 ≈ 15.87th percentile ≈ -1.0
        assert abs(q_low - (-1.0)) < 0.1
        assert abs(q_high - 1.0) < 0.1

    def test_central_abs(self):
        rng = np.random.default_rng(42)
        scores = np.abs(rng.normal(0, 1, size=10000))
        alpha = 0.3173
        q_low, q_high = extract_quantiles(scores, alpha, how="abs", ci_type="central")
        # symmetric: should be roughly ±1
        assert q_low < 0
        assert q_high > 0
        assert abs(q_high + q_low) < 0.2  # symmetric

    def test_invalid_ci_type(self):
        with pytest.raises(ValueError, match="Unknown ci_type"):
            extract_quantiles(np.array([1, 2, 3]), 0.05, ci_type="bad")

    def test_invalid_how(self):
        with pytest.raises(ValueError, match="Unknown how"):
            extract_quantiles(np.array([1, 2, 3]), 0.05, how="bad")


class TestComputeConfidenceInterval:
    def test_basic(self):
        scores = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        y_pred = np.array([5.0, 6.0])
        lower, upper = compute_confidence_interval(
            y_pred, scores, alpha=0.1, how="diff", ci_type="asymmetric"
        )
        assert lower.shape == (2,)
        assert upper.shape == (2,)
        assert np.all(lower < upper)

    def test_scalar(self):
        scores = np.linspace(-2, 2, 100)
        lower, upper = compute_confidence_interval(
            3.0, scores, alpha=0.3173, how="diff", ci_type="asymmetric"
        )
        assert lower < upper


class TestFileBasedCI:
    def test_round_trip(self, tmp_path):
        scores = np.array([-1.0, 0.0, 1.0, 2.0])
        npz_path = tmp_path / "scores.npz"
        np.savez(npz_path, MyModel=scores)

        lower, upper = compute_confidence_intervals_from_file(
            np.array([5.0]),
            npz_path,
            "MyModel",
            alpha=0.3173,
        )
        assert lower.shape == (1,)
        assert np.all(lower < upper)

    def test_missing_model(self, tmp_path):
        npz_path = tmp_path / "scores.npz"
        np.savez(npz_path, A=np.array([1.0]))
        with pytest.raises(KeyError, match="ModelX"):
            compute_confidence_intervals_from_file(
                np.array([1.0]), npz_path, "ModelX", alpha=0.05
            )


class TestSaveQuantiles:
    def test_json_content(self, tmp_path):
        quantiles = {"M1": (-0.5, 0.5), "M2": (-1.0, 1.0)}
        path = save_quantiles(quantiles, 0.3173, "asymmetric", "diff", tmp_path)
        data = json.loads(path.read_text())
        assert data["alpha"] == pytest.approx(0.3173)
        assert data["ci_type"] == "asymmetric"
        assert "M1" in data["quantiles"]
        assert data["quantiles"]["M1"]["q_low"] == pytest.approx(-0.5)


class TestSaveIntervals:
    def test_csv(self, tmp_path):
        import pandas as pd

        df = pd.DataFrame(
            {"model": ["A"], "y_pred": [1.0], "lower": [0.5], "upper": [1.5]}
        )
        path = save_intervals(df, tmp_path)
        assert path.exists()


# ---------------------------------------------------------------------------
# strategies.py — CalibrationResult
# ---------------------------------------------------------------------------


class TestCalibrationResult:
    def test_defaults(self):
        r = CalibrationResult()
        assert r.scores == {}
        assert r.quantiles == {}
        assert r.config is None

    def test_with_data(self):
        cfg = CalibrationConfig(target="mu_hat", alpha=0.05)
        r = CalibrationResult(
            scores={"M": np.array([0.1, 0.2])},
            quantiles={"M": (-0.5, 0.5)},
            config=cfg,
        )
        assert len(r.scores["M"]) == 2
        assert r.config.alpha == 0.05
