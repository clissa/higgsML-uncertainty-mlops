"""Evaluation — performance metrics, calibration quality, and pseudoexperiment analysis."""

from conformal_predictions.evaluation.metrics import (  # noqa: F401
    compute_calibration_metrics,
    compute_ci_score,
    compute_coverage,
    compute_mean_width,
    compute_performance_metrics,
)
from conformal_predictions.evaluation.pseudoexperiments import (  # noqa: F401
    evaluate_on_test_set,
    inference_on_test_set,
)
