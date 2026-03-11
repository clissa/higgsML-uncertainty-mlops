"""W&B artifact helpers for data lineage.

Data artifacts use ``add_reference(file://...)`` so files stay on disk;
model artifacts use ``add_file()`` (uploaded).

All functions return ``None`` when ``wandb_run`` is ``None``, so callers
never need to guard on wandb availability.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Optional wandb — mirrors tracker.py convention
# ---------------------------------------------------------------------------
try:
    import wandb as _wandb  # type: ignore[import]

    _WANDB_AVAILABLE = True
except ImportError:
    _wandb = None  # type: ignore[assignment]
    _WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def artifact_name(dataset: str, mu: float, split: str) -> str:
    """Return the canonical W&B artifact name for a data split.

    Examples
    --------
    >>> artifact_name("toy", 1.0, "train")
    'toy-1.0-train'
    """
    return f"{dataset}-{mu}-{split}"


# ---------------------------------------------------------------------------
# Data artifacts
# ---------------------------------------------------------------------------


def log_or_use_data_artifact(
    wandb_run: object | None,
    name: str,
    files: Sequence[str | Path],
    split_params: dict | None = None,
    version: str = "latest",
) -> object | None:
    """Log a data artifact or declare usage of a pinned version.

    Parameters
    ----------
    wandb_run
        Active ``wandb.Run`` instance, or ``None`` to no-op.
    name
        Artifact name (use :func:`artifact_name`).
    files
        Local file paths to reference.
    split_params
        Optional metadata dict (seed, split sizes, etc.) stored on the
        artifact.
    version
        ``"latest"`` → create/log a new artifact with ``file://``
        references.  Any other value (e.g. ``"v3"``) → call
        ``use_artifact(name:version)`` to declare consumption without
        logging new data.

    Returns
    -------
    wandb.Artifact | None
    """
    if wandb_run is None or not _WANDB_AVAILABLE:
        return None

    try:
        if version == "latest":
            art = _wandb.Artifact(name, type="dataset")
            for f in files:
                art.add_reference(f"file://{Path(f).resolve()}")
            if split_params:
                art.metadata.update(split_params)
            wandb_run.log_artifact(art)
            return art
        else:
            return wandb_run.use_artifact(f"{name}:{version}")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Model artifacts
# ---------------------------------------------------------------------------


def log_model_artifact(
    wandb_run: object | None,
    model_dir: str | Path,
    model_name: str,
) -> object | None:
    """Log a trained model artifact (uploaded to W&B).

    The artifact is named *model_name* (e.g. ``"mlp"``).  W&B handles
    version sequencing automatically (v0, v1, ..., latest).

    Parameters
    ----------
    wandb_run
        Active ``wandb.Run`` instance, or ``None`` to no-op.
    model_dir
        Directory containing the serialised model file(s).
    model_name
        Model name used as the W&B artifact name (e.g. ``"mlp"``).

    Returns
    -------
    wandb.Artifact | None
    """
    if wandb_run is None or not _WANDB_AVAILABLE:
        return None

    model_path = Path(model_dir)

    try:
        art = _wandb.Artifact(model_name, type="model")
        for f in sorted(model_path.iterdir()):
            if f.is_file():
                art.add_file(str(f))
        wandb_run.log_artifact(art)
        return art
    except Exception:
        return None
