# Phase 4.5 — Dashboard Taxonomy Refactor

## Sub-phase A: Naming & Tracker

- [x] 1. Create `mlops/log_keys.py` — `wandb_key()` helper + section constants
- [x] 2. Add `tracker.log_image(key, path)` method
- [x] 3. Add `tracker.log_table(key, wandb_table)` method

## Sub-phase B: New Artefacts

- [x] 4. Store `_X_train_unscaled` in `Trainer.train()`
- [x] 5. Compute & store `_train_predictions` / `_train_pred_labels`
- [x] 6. New `_log_eda()` — class balance scalar + target distribution plot
- [x] 7. `plot_target_distribution()` in `plots.py`
- [x] 8. `plot_predictions_ecdf()` in `plots.py`
- [x] 9. `plot_nonconformity_ecdf()` in `plots.py`
- [x] 10. `plot_nonconformity_by_class()` in `plots.py`
- [x] 11. `plot_distribution()` (generic histogram) in `plots.py`
- [x] 12. `plot_confusion_matrix()` in `plots.py`
- [x] 13. Extend ROC / PR curves with optional train overlay
- [x] 14. Extend `CalibrationResult` — per-block quantiles + calib predictions
- [x] 15. `run_calibration()` populates per-block & calib-pred fields
- [x] 16. `compute_per_example_loss()` in `metrics.py`
- [x] 17. `error_analysis.py` — `build_top_errors_table` / `build_top_errors_wandb_table`
- [x] 18. New `_log_calibration_plots()` in trainer
- [x] 19. New `_log_calibration_metrics()` in trainer
- [x] 20. New `_log_error_analysis()` in trainer

## Sub-phase C: Migrate Keys & Clean Up

- [x] 21. Validation metrics → `Evaluation/val/<metric>`
- [x] 22. Train metrics → `Evaluation/train/<metric>`
- [x] 23. Calibration scalars → `Calibration/…`
- [x] 24. Remove `_log_wandb_image()` — all calls use `tracker.log_image()`
- [x] 25. `reports.py` — grouped sections (EDA / Evaluation / Calibration / ErrorAnalysis)

## Tests

- [x] 26. `test_log_keys.py` — key construction & validation
- [x] 27. `test_evaluation.py` — new plot + metric functions
- [x] 28. `test_calibration.py` — per-block field tests
- [x] 29. `test_tracking.py` — log_image/log_table + key format tests

## Verification

- [x] All 113 tests pass (`pytest tests/ -v`)
- [x] End-to-end pipeline runs cleanly (exit code 0, 49 plots + 32 stat files)
- [x] Zero residual old-style keys in `src/` (grep audit clean)

---

# Phase 4.7 — Single-Model Pipeline Refactor

## Config & Factory

- [x] 1. Add `ModelConfig` dataclass to `config.py` (name, params, validation)
- [x] 2. Add `VALID_MODEL_NAMES` constant
- [x] 3. Add `_build_model_config()` YAML builder (dict, string, and default paths)
- [x] 4. Wire `model: ModelConfig` into `TrainingConfig`
- [x] 5. Wire `_build_model_config()` into `load_training_config()`
- [x] 6. Rewrite `training/models.py` — `build_model()` factory (single-model dict)
- [x] 7. Deprecate `build_default_models()` with warnings

## Trainer & Scripts

- [x] 8. `Trainer.fit()` → uses `build_model(self.config.model, self.config.seed)`
- [x] 9. Simplify `_ref_efficiencies` (direct assignment instead of averaging)
- [x] 10. Add `--model` CLI flag to `run_train.py`
- [x] 11. Print model name at startup in `run_train.py`
- [x] 12. Add `model_name` to `run_metadata.json` output

## Config & Legacy

- [x] 13. Add `model:` section to `configs/train_toy.yaml`
- [x] 14. Add `model:` section to `configs/test_wandb.yaml`
- [x] 15. Update comments in `test_wandb.yaml` (single-model references)
- [x] 16. Add deprecation warnings to legacy `train.py` / `train_higgs.py`
- [x] 17. Export `build_model` from `training/__init__.py`

## Tests

- [x] 18. `test_model_config.py` — ModelConfig validation (valid/invalid names)
- [x] 19. `test_model_config.py` — build_model factory (all families + params)
- [x] 20. `test_model_config.py` — YAML loading (dict, string, missing section)
- [x] 21. All 131 tests pass (`pytest tests/ -v`)

## Verification

- [x] End-to-end MLP pipeline (19 plots, 16 stats, exit 0)
- [x] `--model glm` CLI override works
- [x] `model_name: "mlp"` in `run_metadata.json`
- [x] Zero residual `build_default_models()` calls in `src/` (only re-export + deprecated def)

## Review

Phase 4.7 is commit-ready. All acceptance criteria met.

Option A chosen: single-family-per-run with light multi-family support.
MLP is the default; GLM and Random Forest remain available via config/CLI.
