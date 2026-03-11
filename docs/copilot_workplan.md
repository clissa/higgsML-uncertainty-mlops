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
Dashboard enrichment/consolidation

  ### 4.9-A — W&B Artifact Integration for Data Lineage

  Goal: version raw data and splits as `wandb.Artifact` objects (reference-only, files stay on disk)
  and trained models as uploaded artifacts, so the W&B lineage graph connects data → runs → models.

  **Decisions**
  - Data artifacts use `artifact.add_reference(f"file://{path}")` — W&B computes MD5 checksums,
    deduplicates by digest, files stay on local/EOS filesystem.
  - Model artifacts use `artifact.add_file()` (upload) — sklearn models are KB–MB.
  - Naming: `{dataset}-{mu}-{split}` for data (e.g. `toy-1.0-train`); `{run_id}-{model_name}` for models.
  - Idempotent: W&B won't create a new version when content digest is unchanged.
  - `artifact_version: "latest"` (default) logs fresh artifacts; pinned value (e.g. `"v3"`) skips
    logging and only calls `run.use_artifact(name:v3)`.
  - All artifact logic gated on `wandb_enabled and _WANDB_AVAILABLE` — degrades silently.
  - Evaluation outputs (plots, metric CSVs) continue using the existing local manifest (out of scope).

  #### Phase 1 — Foundation

  **Step 1** — Create `src/conformal_predictions/mlops/artifacts.py` *(new)*
  - `artifact_name(dataset, mu, split)` → canonical name
  - `log_or_use_data_artifact(wandb_run, name, files, split_params, version)` — if `"latest"`:
    creates artifact, adds `file://` references + split metadata, calls `log_artifact()`; if pinned:
    calls `use_artifact(name:version)`. Returns artifact or None.
  - `log_model_artifact(wandb_run, model_dir, run_id, model_name)` — `add_file()` + `log_artifact()`
  - All functions return None when `wandb_run is None`.

  **Step 2** — `TrackingConfig` in `config.py`: add `artifact_version: str = "latest"`;
  update `_build_tracking_config()` to parse it from YAML.

  **Step 3** — `Tracker` in `tracker.py`: add `log_data_artifact()`, `use_data_artifact()`,
  `log_model_artifact()` convenience methods delegating to `artifacts.py`.

  #### Phase 2 — Data Artifacts *(depends on Phase 1)*

  **Step 4** — `Trainer.load_data()`: after `list_split_files()` returns, log raw dataset artifact
  (all `.npz` files from `mu_dir`) and one split artifact per split (train/val/calib/test) with
  metadata containing seed, valid_size, calib_size, test_prefixes.

  **Step 5** — `Trainer.calibrate()` and `Trainer.evaluate()`: call `tracker.use_data_artifact()`
  at the start of each stage to declare consumption of calib/test artifacts (lineage graph).

  #### Phase 3 — Model Artifacts *(depends on Phase 1, parallel with Phase 2)*

  **Step 6** — `Trainer.train()`: after `self.fit()`, serialize each model + scaler via
  `joblib.dump()` → `{output_dir}/models/{model_name}.joblib`, register in local manifest,
  call `tracker.log_model_artifact()`.

  #### Phase 4 — Tests *(parallel with Phases 2–3)*

  **Step 7** — Create `tests/test_artifacts.py` *(new)*
  - `artifact_name()` naming convention
  - `log_or_use` with `"latest"` → `log_artifact` called with correct file references
  - `log_or_use` with pinned version → `use_artifact` called
  - No-op when `wandb_run is None`
  - Split metadata included in artifact
  - Model artifact uses `add_file`
  - `artifact_version` parsed from YAML

  #### Phase 5 — Config & Documentation

  **Step 8** — Add `artifact_version: "latest"` under `tracking:` in `configs/test_wandb.yaml`
  and `configs/train_toy.yaml`; fix the comment in `test_wandb.yaml` that overstates implemented
  artifact behaviour.

  **Relevant files**:
  `mlops/artifacts.py` (new), `config.py`, `mlops/tracker.py`, `training/trainer.py`,
  `tests/test_artifacts.py` (new), `configs/test_wandb.yaml`, `configs/train_toy.yaml`

  **Verification**:
  1. `pytest tests/test_artifacts.py -v` — all new artifact tests pass
  2. `pytest tests/ -v` — no regressions
  3. Smoke (wandb disabled): pipeline completes, model serialised, no artifact errors
  4. Smoke (wandb enabled): W&B UI shows `toy-1.0-{raw,train,val,calib,test}` + `{run_id}-mlp`;
     lineage graph connects raw → run → splits + model
  5. Version pinning: `artifact_version: "v0"` → `use_artifact` called, no new versions
  6. Idempotency: two identical runs → same digest, no duplicate versions

Phase 5
Sweep

Phase 6
Registry

Phase 7
Docs + tests

The workplan defines the implementation order.
Each phase should result in a working, runnable pipeline.
Do not break previously implemented functionality.