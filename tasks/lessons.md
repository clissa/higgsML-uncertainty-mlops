# Lessons Learned

## Phase 4.5 — Dashboard Taxonomy Refactor

### Pattern: Centralise magic strings early
- **Mistake risk**: Scattered string literals for wandb keys are fragile and hard to audit.
- **Rule**: Always route logging keys through a single helper (`wandb_key()`) with a whitelist of allowed sections.

### Pattern: Fail-silent tracker methods
- **Lesson**: `tracker.log_image()` and `tracker.log_table()` must silently no-op when wandb is unavailable, just like `tracker.log()`.
- **Rule**: Every tracker method should guard on `_wandb_run is not None`.

### Pattern: Keep plot functions pure
- **Lesson**: Plot functions should accept data arrays and return `(fig, ax)`. Side effects (saving, logging) belong in the caller.
- **Rule**: Never call `tracker.log_image()` inside `plots.py`.

### Pattern: Dataclass defaults for backward compatibility
- **Lesson**: When extending a dataclass (e.g., `CalibrationResult`), new fields must have defaults (`None`, `field(default_factory=dict)`) so existing callers don't break.
- **Rule**: Always add new dataclass fields with defaults at the end.

### Pattern: Use `create_file` only for new files
- **Lesson**: `create_file` fails if the file already exists. Use `replace_string_in_file` to update existing files.
- **Rule**: Check if a file exists before choosing create vs. edit.

## Phase 4.7 — Single-Model Pipeline Refactor

### Pattern: Dict[str, model] is a natural single-model container
- **Lesson**: When all downstream code iterates `Dict[str, model].items()`, switching from 3 entries to 1 entry requires zero loop refactoring. The single-entry dict pattern avoids touching ~20 function signatures.
- **Rule**: Prefer narrowing the dict contents over changing the dict type when simplifying multiplicity.

### Pattern: Config factory over model registry
- **Lesson**: A simple factory function (`build_model()`) with a name→class mapping is sufficient for 3 model families. No need for a plugin system, registry, or metaclass pattern.
- **Rule**: Prefer explicit factory functions over generic registration for small model catalogues.

### Pattern: Validate config eagerly in __post_init__
- **Lesson**: Frozen dataclass `__post_init__` is the right place to validate enum-like fields (e.g., model name). Errors surface at config load time, not at training time.
- **Rule**: Use `__post_init__` for validation in frozen dataclasses; raise clear `ValueError` with valid options listed.

## Bug Fix — mu_hat test-set bias (2026-03-11)

### Pattern: Pass derived state through all pipeline stages
- **Bug**: `Trainer.evaluate()` called `evaluate_on_test_set()` without `ref_efficiencies_dict`, causing it to default to `(1.0, 1.0)`. Calibration stage passed them correctly, so calibration mu_hat ≈ 1.0 but test mu_hat ≈ -0.94.
- **Root cause**: When `evaluate()` was wired up, the `ref_efficiencies_dict` kwarg was omitted from the call despite `self._ref_efficiencies` being available.
- **Rule**: When a downstream function has an Optional parameter with a fallback default (e.g., `ref_efficiencies_dict=None` → `{name: (1.0, 1.0)}`), always pass the real value explicitly. Silent defaults on physics quantities are dangerous.
- **Detection**: The calibration mu_hat was correct (~1.0) while test mu_hat was biased (~-0.94). Comparing calibration vs test mu_hat distributions immediately reveals the inconsistency.

## Fix W&B Artifact Lineage

### Pattern: W&B lineage requires separate runs for multi-step graphs
- **Bug**: All artifacts (raw data, splits, model) were logged as outputs of a single W&B run, producing a flat lineage graph instead of `raw → splits → model`.
- **Root cause**: `log_artifact()` and `use_artifact()` on the same artifact within the same run doesn't create the expected input→output edges. W&B needs distinct runs for each pipeline stage.
- **Rule**: For multi-step lineage in W&B, create dedicated short-lived runs with appropriate `job_type` labels (e.g., "dataset-logging", "dataset-splitting"). Each run should produce outputs and declare inputs via `use_artifact`, so W&B can draw edges between stages.
- **Pattern**: Helper runs must be finished (`run.finish()`) before the main training run starts (`wandb.init`). Use `art.wait()` on upstream artifacts to ensure server-side availability.
