Phase 1
Refactor training and config system
**Status: DONE**

Phase 1 deliverables:
- `src/conformal_predictions/training/` package (core.py, models.py, trainer.py)
- `src/conformal_predictions/config.py` — TrainingConfig with YAML loader
- `src/conformal_predictions/mlops/run_context.py` — lightweight RunContext
- `scripts/run_train.py` — config-driven CLI entrypoint
- `configs/train_toy.yaml` — training config for toy pipeline
- All 8 tests pass, backward-compatible with existing scripts

Phase 2
Evaluation primitives, calibration, and metrics
**Status: DONE**

Phase 2 deliverables:
- `src/conformal_predictions/calibration/` package (scores.py, intervals.py, strategies.py)
- `src/conformal_predictions/evaluation/` package (metrics.py, pseudoexperiments.py)
- `CalibrationConfig` and `EvaluationConfig` frozen dataclasses in config.py
- Trainer decomposed into `train()`, `calibrate()`, `evaluate()` stages
- `--mode` flag on `run_train.py` (train, calibrate, evaluate, train+calibrate, all)
- 7 performance metrics (loss, accuracy, precision, recall, f1, pr_auc, roc_auc)
- 3 calibration quality metrics (coverage, width, ci_score)
- Nonconformity scores parameterised by alpha and ci_type
- Full artifact persistence (JSON, CSV, NPZ, PNG)
- 30 new unit tests; all 46 tests pass
- Full backward compatibility with Phase 1 scripts

Phase 3
Tracking + artifacts

Phase 4
Evaluation + plots/reports

Phase 4.5
Refactor wandb logging to enforce dashboard structure
**Status: DONE**

Phase 4.7
Single model refactor: pipeline runs one model at a time (MLP default)
**Status: DONE**

Phase 4.7 deliverables:
- `ModelConfig` dataclass in `config.py` with validation
- `build_model()` factory in `training/models.py` (single-model dict)
- `Trainer.fit()` uses config-driven model selection
- `--model` CLI flag on `run_train.py`
- `model_name` in `run_metadata.json`
- `model:` section in YAML configs (MLP default)
- `build_default_models()` deprecated with warnings
- Legacy scripts annotated with deprecation
- 18 new tests in `test_model_config.py`; all 131 tests pass
- End-to-end pipeline verified (single model, 19 plots, 16 stats)

Phase 4.9
Dashboard enrichment/consolitation

Phase 5
Sweep

Phase 6
Registry

Phase 7
Docs + tests

The workplan defines the implementation order.
Each phase should result in a working, runnable pipeline.
Do not break previously implemented functionality.