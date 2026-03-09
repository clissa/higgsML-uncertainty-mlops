"""Lightweight run-context abstraction.

Captures core metadata for every training run so that results are
traceable and reproducible without requiring a full experiment tracker.

Usage::

    from conformal_predictions.config import load_training_config
    from conformal_predictions.mlops.run_context import RunContext

    cfg = load_training_config("configs/train_toy.yaml")
    ctx = RunContext.create(cfg, config_path="configs/train_toy.yaml")
    print(ctx.run_id, ctx.output_dir)
    # ... run training ...
    ctx.save_metadata()   # writes run_metadata.json into output_dir
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from conformal_predictions.config import TrainingConfig


def _get_git_commit() -> Optional[str]:
    """Return the current git commit hash, or *None* if unavailable."""
    try:
        import git

        repo = git.Repo(search_parent_directories=True)
        return repo.head.commit.hexsha
    except Exception:
        return None


@dataclass
class RunContext:
    """Metadata for a single training run.

    Attributes
    ----------
    run_id : str
        Unique identifier (short UUID).
    timestamp : str
        ISO-8601 UTC timestamp of run creation.
    config_snapshot : dict
        Frozen copy of the ``TrainingConfig`` used for this run.
    config_path : str | None
        Filesystem path to the config YAML (if loaded from file).
    dataset : str
        Dataset identifier (e.g. ``"toy"``, ``"higgs"``).
    git_commit : str | None
        Git commit hash at the time of the run.
    output_dir : Path
        Root directory for all run artifacts.
    artifacts : list
        Registry of saved files populated by :meth:`save_artifact`.
    """

    run_id: str = ""
    timestamp: str = ""
    config_snapshot: dict = field(default_factory=dict)
    config_path: Optional[str] = None
    dataset: str = ""
    git_commit: Optional[str] = None
    output_dir: Path = Path("results")

    # Phase 3: Artifact manifest — mutable registry of saved files.
    artifacts: list = field(default_factory=list, compare=False)

    @classmethod
    def create(
        cls,
        config: TrainingConfig,
        config_path: Optional[str] = None,
    ) -> "RunContext":
        """Factory that auto-populates run metadata from a config."""
        run_id = uuid.uuid4().hex[:8]
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        run_name = config.run_name or f"run-{run_id}"
        out = Path(config.output_dir) / run_name
        return cls(
            run_id=run_id,
            timestamp=ts,
            config_snapshot=config.to_dict(),
            config_path=str(config_path) if config_path else None,
            dataset=config.dataset,
            git_commit=_get_git_commit(),
            output_dir=out,
        )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def plots_dir(self) -> Path:
        return self.output_dir / "plots"

    @property
    def stats_dir(self) -> Path:
        return self.output_dir / "stats"

    def ensure_dirs(self) -> None:
        """Create the run output directories (plots/ and stats/)."""
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.stats_dir.mkdir(parents=True, exist_ok=True)

    def save_artifact(
        self,
        path: "str | Path",
        type: str,
        format: str,
        description: str = "",
    ) -> None:
        """Register a saved file in the artifact manifest.

        Parameters
        ----------
        path :
            Path **relative** to ``output_dir``.
        type :
            Category e.g. ``"metric"``, ``"plot"``, ``"calibration"``,
            ``"model"``, ``"scores"``.
        format :
            File format e.g. ``"json"``, ``"csv"``, ``"npz"``, ``"png"``.
        description :
            Short human-readable description.
        """
        self.artifacts.append(
            {
                "path": str(path),
                "type": type,
                "format": format,
                "description": description,
            }
        )

    def save_manifest(self, path: Optional[Path] = None) -> Path:
        """Persist the artifact manifest as JSON.

        Returns the path to the written file.
        """
        path = path or (self.output_dir / "artifact_manifest.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump({"run_id": self.run_id, "artifacts": self.artifacts}, fh, indent=2)
        return path

    def save_metadata(self, path: Optional[Path] = None) -> Path:
        """Persist run metadata as JSON and write the artifact manifest.

        Returns the path to the written metadata file.
        """
        path = path or (self.output_dir / "run_metadata.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "config_snapshot": self.config_snapshot,
            "config_path": self.config_path,
            "dataset": self.dataset,
            "git_commit": self.git_commit,
            "output_dir": str(self.output_dir),
        }
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)
        # Always write the artifact manifest alongside run_metadata.json
        self.save_manifest()
        return path