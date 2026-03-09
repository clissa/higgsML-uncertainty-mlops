"""Training package — scientific computation and model utilities."""

# Re-export all public symbols from core so that existing imports
# ``from conformal_predictions.training import ...`` keep working.
from conformal_predictions.training.core import (  # noqa: F401
    compute_confidence_interval,
    compute_mu_hat,
    compute_nonconformity_scores,
    evaluate_models,
    get_events_count,
    inference_on_test_set,
    list_split_files,
)
from conformal_predictions.training.models import build_default_models  # noqa: F401
