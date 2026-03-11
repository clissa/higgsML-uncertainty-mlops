"""W&B artifact helpers for data lineage.

Data artifacts use ``add_reference(file://...)`` so files stay on disk;
model artifacts use ``add_file()`` (uploaded).

All functions return ``None`` when ``wandb_run`` is ``None``, so callers
never need to guard on wandb availability.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Sequence

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

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
# Data-lineage helper runs
# ---------------------------------------------------------------------------


def log_raw_data_run(
    project: str,
    entity: str | None,
    name: str,
    files: Sequence[str | Path],
    metadata: dict | None = None,
) -> object | None:
    """Create a short-lived W&B run that logs a raw dataset artifact.

    This produces a "dataset-logging" node in the W&B lineage graph.
    The run is finished before returning so it does not interfere with
    the main training run.

    Parameters
    ----------
    project : str
        W&B project name.
    entity : str | None
        W&B entity (team/user).
    name : str
        Artifact name (e.g. ``"toy-1.0-raw"``).
    files : Sequence[str | Path]
        Local file paths to reference in the artifact.
    metadata : dict | None
        Optional metadata dict stored on the artifact.

    Returns
    -------
    wandb.Artifact | None
    """
    if not _WANDB_AVAILABLE:
        return None
    try:
        run = _wandb.init(
            project=project,
            entity=entity or None,
            name="dataset-logging",
            job_type="dataset-logging",
            reinit=True,
        )
        art = _wandb.Artifact(name, type="dataset")
        for f in files:
            art.add_reference(f"file://{Path(f).resolve()}")
        if metadata:
            art.metadata.update(metadata)
        run.log_artifact(art)
        art.wait()
        run.finish()
        return art
    except Exception:
        logger.debug("log_raw_data_run failed for %s", name, exc_info=True)
        return None


def log_split_data_run(
    project: str,
    entity: str | None,
    raw_artifact_name: str,
    splits: Dict[str, Sequence[str | Path]],
    metadata: dict | None = None,
) -> Dict[str, object] | None:
    """Create a short-lived W&B run that splits raw data into partitions.

    The run declares the raw artifact as an input (``use_artifact``) and
    logs one output artifact per split, producing a "dataset-splitting"
    node in the W&B lineage graph.

    Parameters
    ----------
    project : str
        W&B project name.
    entity : str | None
        W&B entity (team/user).
    raw_artifact_name : str
        Name of the raw dataset artifact (e.g. ``"toy-1.0-raw"``).
        Will be fetched as ``<name>:latest``.
    splits : Dict[str, Sequence[str | Path]]
        Mapping of split artifact name → list of local file paths.
        Example: ``{"toy-1.0-train": [f1, f2], "toy-1.0-val": [f3]}``.
    metadata : dict | None
        Optional metadata dict attached to every split artifact.

    Returns
    -------
    dict[str, wandb.Artifact] | None
    """
    if not _WANDB_AVAILABLE:
        return None
    try:
        run = _wandb.init(
            project=project,
            entity=entity or None,
            name="dataset-splitting",
            job_type="dataset-splitting",
            reinit=True,
        )
        # Declare the raw dataset as input → lineage edge
        run.use_artifact(f"{raw_artifact_name}:latest")

        result: Dict[str, object] = {}
        for split_name, files in splits.items():
            art = _wandb.Artifact(split_name, type="dataset")
            for f in files:
                art.add_reference(f"file://{Path(f).resolve()}")
            if metadata:
                art.metadata.update(metadata)
            run.log_artifact(art)
            result[split_name] = art

        run.finish()
        return result
    except Exception:
        logger.debug(
            "log_split_data_run failed for %s", raw_artifact_name, exc_info=True
        )
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
