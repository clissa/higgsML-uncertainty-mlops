"""Lightweight run tracker.

Accumulates scalar metrics from all pipeline stages, flushes them to
``metrics.json`` at run end, appends a summary record to the local run
index, and (optionally) forwards everything to wandb.

Usage::

    from conformal_predictions.mlops.tracker import Tracker

    tracker = Tracker(ctx, cfg.tracking)
    tracker.start(cfg.to_dict())

    tracker.log("Evaluation/val/f1", 0.82, stage="train")
    tracker.log("Calibration/metrics/coverage", 0.68, stage="evaluate")

    tracker.finish()   # writes metrics.json, appends to runs_index.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:
    from conformal_predictions.config import TrackingConfig
    from conformal_predictions.mlops.run_context import RunContext

# ---------------------------------------------------------------------------
# Optional wandb — silently skip if not installed
# ---------------------------------------------------------------------------
try:
    import wandb as _wandb  # type: ignore[import]

    _WANDB_AVAILABLE = True
except ImportError:
    _wandb = None  # type: ignore[assignment]
    _WANDB_AVAILABLE = False


class Tracker:
    """Accumulates metrics and flushes them to ``metrics.json`` at run end.

    Parameters
    ----------
    run_ctx : RunContext
        The current run context (provides output dir, run_id, etc.).
    config : TrackingConfig
        Tracking configuration (enabled flags, wandb settings, index path).
    """

    def __init__(self, run_ctx: "RunContext", config: "TrackingConfig") -> None:
        self._ctx = run_ctx
        self._config = config
        self._metrics: list[dict] = []
        self._wandb_run = None
        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, config_snapshot: dict) -> None:
        """Initialise the tracker (and optionally wandb).

        Parameters
        ----------
        config_snapshot : dict
            Serialisable snapshot of the training config, forwarded to
            ``wandb.init(config=...)`` when wandb is enabled.
        """
        self._metrics = []
        self._started = True

        if self._config.wandb_enabled and _WANDB_AVAILABLE:
            try:
                self._wandb_run = _wandb.init(
                    project=self._config.wandb_project,
                    entity=self._config.wandb_entity or None,
                    name=self._ctx.run_id,
                    config=config_snapshot,
                    reinit=True,
                )
            except Exception:
                self._wandb_run = None

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
        stage: Optional[str] = None,
    ) -> None:
        """Log a scalar metric.

        Parameters
        ----------
        name : str
            Metric name, e.g. ``"Evaluation/val/f1"``.
        value : float
            Scalar value.
        step : int, optional
            Training step or epoch number (forwarded to wandb).
        stage : str, optional
            Pipeline stage label e.g. ``"train"``, ``"calibrate"``,
            ``"evaluate"``.
        """
        record = {
            "name": name,
            "value": float(value),
            "step": step,
            "stage": stage,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self._metrics.append(record)

        if self._wandb_run is not None:
            try:
                log_dict: dict = {name: float(value)}
                kwargs: dict = {}
                if step is not None:
                    kwargs["step"] = step
                self._wandb_run.log(log_dict, **kwargs)
            except Exception:
                pass

    def log_dict(
        self,
        metrics: dict,
        step: Optional[int] = None,
        stage: Optional[str] = None,
    ) -> None:
        """Log a batch of scalar metrics in a single wandb call.

        All entries are recorded individually in ``self._metrics`` for
        ``metrics.json``, but forwarded to wandb as one
        ``wandb_run.log(dict, step=step)`` call so that metrics logged at
        the same step are grouped correctly in the wandb UI.

        Parameters
        ----------
        metrics : dict
            Mapping of metric name → scalar value.
        step : int, optional
            Training step / epoch number.
        stage : str, optional
            Pipeline stage label, e.g. ``"train"``.
        """
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        wandb_payload: dict = {}
        for name, value in metrics.items():
            record = {
                "name": name,
                "value": float(value),
                "step": step,
                "stage": stage,
                "timestamp": ts,
            }
            self._metrics.append(record)
            wandb_payload[name] = float(value)

        if self._wandb_run is not None and wandb_payload:
            try:
                kwargs: dict = {}
                if step is not None:
                    kwargs["step"] = step
                self._wandb_run.log(wandb_payload, **kwargs)
            except Exception:
                pass

    def log_image(self, key: str, path: "Path") -> None:
        """Log a PNG image to wandb under *key*; fails silently.

        Parameters
        ----------
        key : str
            Wandb key, e.g. ``"Evaluation/plots/roc_curve"``.
        path : Path
            Local path to the image file.
        """
        if self._wandb_run is None:
            return
        try:
            self._wandb_run.log({key: _wandb.Image(str(path))})
        except Exception:
            pass

    def log_table(self, key: str, table: object) -> None:
        """Log a ``wandb.Table`` to wandb under *key*; fails silently.

        Parameters
        ----------
        key : str
            Wandb key, e.g. ``"ErrorAnalysis/train/top_errors"``.
        table : wandb.Table
            A wandb Table object.
        """
        if self._wandb_run is None:
            return
        try:
            self._wandb_run.log({key: table})
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Finish
    # ------------------------------------------------------------------

    def log_data_artifact(
        self,
        name: str,
        files: "Sequence[str | Path]",
        split_params: dict | None = None,
    ) -> object | None:
        """Log a data artifact (reference-only) via :mod:`artifacts`.

        Uses ``self._config.artifact_version`` to decide between
        logging a new artifact or declaring usage of a pinned one.
        """
        from conformal_predictions.mlops.artifacts import log_or_use_data_artifact

        return log_or_use_data_artifact(
            self._wandb_run,
            name,
            files,
            split_params=split_params,
            version=self._config.artifact_version,
        )

    def use_data_artifact(self, name: str, version: str | None = None) -> object | None:
        """Declare consumption of an existing data artifact (lineage)."""
        if self._wandb_run is None:
            return None
        ver = version or self._config.artifact_version
        try:
            return self._wandb_run.use_artifact(f"{name}:{ver}")
        except Exception:
            return None

    def log_model_artifact(
        self,
        model_dir: "str | Path",
        model_name: str,
    ) -> object | None:
        """Log a trained model artifact (uploaded) via :mod:`artifacts`."""
        from conformal_predictions.mlops.artifacts import log_model_artifact

        return log_model_artifact(self._wandb_run, model_dir, model_name)

    # ------------------------------------------------------------------

    def finish(self) -> None:
        """Flush metrics, append to run index, and finish wandb.

        Writes ``metrics.json`` to the run output directory, builds a
        run index record, appends it to ``runs_index.json``, and calls
        ``wandb.finish()`` if wandb is active.

        Safe to call even when ``tracking.enabled`` is ``False`` — the
        method becomes a no-op in that case.
        """
        if not self._config.enabled:
            return

        ctx = self._ctx

        # ---- write metrics.json ----
        metrics_path = ctx.output_dir / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as fh:
            json.dump({"run_id": ctx.run_id, "metrics": self._metrics}, fh, indent=2)

        # Register metrics.json in the artifact manifest
        ctx.save_artifact(
            "metrics.json",
            type="metric",
            format="json",
            description="All scalar metrics from all pipeline stages",
        )

        # ---- flat metric summary for the index record ----
        metric_summary: dict = {}
        for entry in self._metrics:
            metric_summary[entry["name"]] = entry["value"]

        # ---- build run index record ----
        model_name = ctx.config_snapshot.get("model", {}).get("name", "unknown")
        record: dict = {
            "run_id": ctx.run_id,
            "timestamp": ctx.timestamp,
            "model_name": model_name,
            "dataset": ctx.dataset,
            "config_path": ctx.config_path,
            "output_dir": str(ctx.output_dir),
            "git_commit": ctx.git_commit,
            "metrics": metric_summary,
        }

        # ---- finish wandb and capture metadata ----
        if self._wandb_run is not None:
            try:
                self._wandb_run.finish()
                record["wandb"] = {
                    "run_id": self._wandb_run.id,
                    "project": self._wandb_run.project,
                    "url": self._wandb_run.url,
                    "synced": True,
                }
            except Exception:
                record["wandb"] = {"synced": False}
            self._wandb_run = None

        # ---- append to local run index ----
        from conformal_predictions.mlops.run_index import append_run

        append_run(record, self._config.index_path)
