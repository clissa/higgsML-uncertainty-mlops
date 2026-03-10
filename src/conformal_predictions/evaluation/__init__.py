"""Evaluation — performance metrics, calibration quality, and pseudoexperiment analysis."""

from conformal_predictions.evaluation.metrics import (  # noqa: F401
    compute_calibration_metrics,
    compute_ci_score,
    compute_coverage,
    compute_mean_width,
    compute_performance_metrics,
)
from conformal_predictions.evaluation.plots import (  # noqa: F401
    plot_ci_coverage,
    plot_ci_width_distribution,
    plot_confusion_matrix,
    plot_distribution,
    plot_mu_hat_distribution,
    plot_nonconformity_by_class,
    plot_nonconformity_ecdf,
    plot_nonconformity_scores,
    plot_pr_curve,
    plot_predictions_ecdf,
    plot_roc_curve,
    plot_target_distribution,
)

# NOTE: pseudoexperiments is intentionally NOT imported here to avoid a circular
# import: calibration.strategies → data_viz → evaluation → pseudoexperiments →
# calibration.strategies.  Import directly from the submodule when needed:
#   from conformal_predictions.evaluation.pseudoexperiments import evaluate_on_test_set
from conformal_predictions.evaluation.reports import generate_run_report  # noqa: F401
