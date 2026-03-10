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
