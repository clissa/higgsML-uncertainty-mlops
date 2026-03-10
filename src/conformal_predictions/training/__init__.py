"""Training package — scientific computation and model utilities.

.. note:: Phase 2 migration
   Calibration functions (``compute_nonconformity_scores``,
   ``compute_confidence_interval``, ``compute_mu_hat``) now have
   canonical implementations in :mod:`conformal_predictions.calibration`.
   Evaluation helpers (``evaluate_models``, ``inference_on_test_set``)
   have canonical implementations in :mod:`conformal_predictions.evaluation`.

   The re-exports below are kept so that existing imports
   ``from conformal_predictions.training import ...`` continue to work.
"""

# Re-export all public symbols from core so that existing imports keep working.
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
from conformal_predictions.training.models import build_model  # noqa: F401
