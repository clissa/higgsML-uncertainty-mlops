"""Tests for Phase 3 tracking components.

Covers:
- RunContext artifact manifest: save_artifact(), save_manifest(), and
  save_metadata() writing the manifest alongside run_metadata.json.
- run_index: append_run(), load_index(), atomic writes.
- Tracker: metric accumulation, metrics.json flush, run index append, and
  wandb no-op when not installed.
- TrackingConfig: YAML loading.
"""

from __future__ import annotations

import json
import sys
import types
import uuid
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from conformal_predictions.config import TrackingConfig, load_training_config
from conformal_predictions.mlops.run_context import RunContext
from conformal_predictions.mlops.run_index import append_run, load_index
from conformal_predictions.mlops.tracker import Tracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(tmp_path: Path) -> RunContext:
    """Return a minimal RunContext pointing at *tmp_path*."""
    return RunContext(
        run_id=uuid.uuid4().hex[:8],
        timestamp="2026-01-01T00:00:00+00:00",
        config_snapshot={},
        config_path=None,
        dataset="toy",
        git_commit=None,
        output_dir=tmp_path / "run-test",
    )


def _make_tracking_config(**kwargs) -> TrackingConfig:
    return TrackingConfig(**kwargs)


# ---------------------------------------------------------------------------
# A) Artifact manifest
# ---------------------------------------------------------------------------


class TestArtifactManifest:
    def test_save_artifact_appends(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        assert ctx.artifacts == []

        ctx.save_artifact("stats/foo.csv", type="metric", format="csv", description="test")
        assert len(ctx.artifacts) == 1
        a = ctx.artifacts[0]
        assert a["path"] == "stats/foo.csv"
        assert a["type"] == "metric"
        assert a["format"] == "csv"
        assert a["description"] == "test"

    def test_save_artifact_accumulates(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        ctx.save_artifact("stats/a.json", type="metric", format="json")
        ctx.save_artifact("plots/b.png", type="plot", format="png")
        assert len(ctx.artifacts) == 2

    def test_save_manifest_writes_json(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        ctx.save_artifact("stats/scores.npz", type="calibration", format="npz", description="scores")
        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = ctx.save_manifest()

        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text())
        assert data["run_id"] == ctx.run_id
        assert len(data["artifacts"]) == 1
        assert data["artifacts"][0]["path"] == "stats/scores.npz"

    def test_save_metadata_also_writes_manifest(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        ctx.save_artifact("stats/metrics.json", type="metric", format="json")
        meta_path = ctx.save_metadata()

        assert meta_path.exists()
        manifest_path = ctx.output_dir / "artifact_manifest.json"
        assert manifest_path.exists()

        data = json.loads(manifest_path.read_text())
        assert data["run_id"] == ctx.run_id
        assert any(a["path"] == "stats/metrics.json" for a in data["artifacts"])

    def test_artifacts_stored_as_relative_paths(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        ctx.save_artifact("stats/calibration_summary.json", type="calibration", format="json")
        assert ctx.artifacts[0]["path"] == "stats/calibration_summary.json"

    def test_default_description_is_empty(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        ctx.save_artifact("x.csv", type="metric", format="csv")
        assert ctx.artifacts[0]["description"] == ""


# ---------------------------------------------------------------------------
# B) Run index
# ---------------------------------------------------------------------------


class TestRunIndex:
    def _record(self, run_id: str = "abc12345") -> dict:
        return {
            "run_id": run_id,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "dataset": "toy",
            "config_path": None,
            "output_dir": "results/run-abc12345",
            "git_commit": None,
            "metrics": {"logistic_regression.val_f1": 0.82},
        }

    def test_append_creates_file(self, tmp_path):
        index = tmp_path / "runs_index.json"
        append_run(self._record(), index)
        assert index.exists()
        data = json.loads(index.read_text())
        assert len(data) == 1
        assert data[0]["run_id"] == "abc12345"

    def test_append_does_not_overwrite(self, tmp_path):
        index = tmp_path / "runs_index.json"
        append_run(self._record("run1"), index)
        append_run(self._record("run2"), index)
        data = json.loads(index.read_text())
        assert len(data) == 2
        assert {r["run_id"] for r in data} == {"run1", "run2"}

    def test_load_index_returns_list(self, tmp_path):
        index = tmp_path / "idx.json"
        append_run(self._record("r1"), index)
        append_run(self._record("r2"), index)
        records = load_index(index)
        assert len(records) == 2

    def test_load_index_missing_file(self, tmp_path):
        records = load_index(tmp_path / "nonexistent.json")
        assert records == []

    def test_load_index_corrupt_file(self, tmp_path):
        index = tmp_path / "bad.json"
        index.write_text("not valid json {{{")
        records = load_index(index)
        assert records == []

    def test_append_creates_parent_dirs(self, tmp_path):
        index = tmp_path / "nested" / "dir" / "runs_index.json"
        append_run(self._record(), index)
        assert index.exists()

    def test_append_run_preserves_metrics_field(self, tmp_path):
        index = tmp_path / "idx.json"
        rec = self._record()
        rec["metrics"] = {"foo.bar": 1.23}
        append_run(rec, index)
        loaded = load_index(index)[0]
        assert loaded["metrics"]["foo.bar"] == pytest.approx(1.23)


# ---------------------------------------------------------------------------
# C) Tracker
# ---------------------------------------------------------------------------


class TestTracker:
    def test_log_accumulates_metrics(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        cfg = _make_tracking_config(index_path=str(tmp_path / "idx.json"))
        t = Tracker(ctx, cfg)
        t.start({})
        t.log("model.val_f1", 0.82, stage="train")
        t.log("model.coverage", 0.68, stage="evaluate")
        assert len(t._metrics) == 2
        assert t._metrics[0]["name"] == "model.val_f1"
        assert t._metrics[0]["value"] == pytest.approx(0.82)
        assert t._metrics[0]["stage"] == "train"

    def test_finish_writes_metrics_json(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        ctx.output_dir.mkdir(parents=True)
        cfg = _make_tracking_config(index_path=str(tmp_path / "idx.json"))
        t = Tracker(ctx, cfg)
        t.start({})
        t.log("model.val_f1", 0.82, stage="train")
        t.finish()

        metrics_path = ctx.output_dir / "metrics.json"
        assert metrics_path.exists()
        data = json.loads(metrics_path.read_text())
        assert data["run_id"] == ctx.run_id
        assert len(data["metrics"]) == 1
        assert data["metrics"][0]["name"] == "model.val_f1"

    def test_finish_appends_to_run_index(self, tmp_path):
        index_path = tmp_path / "idx.json"
        ctx = _make_ctx(tmp_path)
        ctx.output_dir.mkdir(parents=True)
        cfg = _make_tracking_config(index_path=str(index_path))
        t = Tracker(ctx, cfg)
        t.start({})
        t.log("x", 1.0, stage="train")
        t.finish()

        records = load_index(index_path)
        assert len(records) == 1
        assert records[0]["run_id"] == ctx.run_id
        assert records[0]["metrics"]["x"] == pytest.approx(1.0)

    def test_finish_registers_metrics_json_in_manifest(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        ctx.output_dir.mkdir(parents=True)
        cfg = _make_tracking_config(index_path=str(tmp_path / "idx.json"))
        t = Tracker(ctx, cfg)
        t.start({})
        t.finish()

        assert any(a["path"] == "metrics.json" for a in ctx.artifacts)

    def test_disabled_tracking_is_noop(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        ctx.output_dir.mkdir(parents=True)
        index_path = tmp_path / "idx.json"
        cfg = _make_tracking_config(enabled=False, index_path=str(index_path))
        t = Tracker(ctx, cfg)
        t.start({})
        t.log("x", 1.0)
        t.finish()

        assert not (ctx.output_dir / "metrics.json").exists()
        assert not index_path.exists()

    def test_multiple_finish_calls_append_index(self, tmp_path):
        """Two separate trackers each call finish → index has 2 records."""
        index_path = tmp_path / "idx.json"
        for i in range(2):
            ctx = RunContext(
                run_id=f"run{i:04d}",
                timestamp="2026-01-01T00:00:00+00:00",
                config_snapshot={},
                dataset="toy",
                output_dir=tmp_path / f"run-run{i:04d}",
            )
            ctx.output_dir.mkdir(parents=True)
            cfg = _make_tracking_config(index_path=str(index_path))
            t = Tracker(ctx, cfg)
            t.start({})
            t.finish()
        records = load_index(index_path)
        assert len(records) == 2


# ---------------------------------------------------------------------------
# D) wandb no-op when not installed
# ---------------------------------------------------------------------------


class TestWandbNoOp:
    def test_wandb_import_error_is_silently_skipped(self, tmp_path):
        """When wandb is not importable, Tracker still runs without error."""
        ctx = _make_ctx(tmp_path)
        ctx.output_dir.mkdir(parents=True)
        cfg = _make_tracking_config(
            wandb_enabled=True,
            index_path=str(tmp_path / "idx.json"),
        )

        # Temporarily hide wandb from the tracker module
        import conformal_predictions.mlops.tracker as tracker_mod

        original_available = tracker_mod._WANDB_AVAILABLE
        original_wandb = tracker_mod._wandb
        try:
            tracker_mod._WANDB_AVAILABLE = False
            tracker_mod._wandb = None
            t = Tracker(ctx, cfg)
            t.start({"key": "value"})
            t.log("model.val_f1", 0.9, stage="train")
            t.finish()
        finally:
            tracker_mod._WANDB_AVAILABLE = original_available
            tracker_mod._wandb = original_wandb

        # pipeline ran: metrics.json and index should exist
        assert (ctx.output_dir / "metrics.json").exists()
        records = load_index(tmp_path / "idx.json")
        assert len(records) == 1
        assert "wandb" not in records[0]

    def test_wandb_disabled_produces_no_wandb_key_in_index(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        ctx.output_dir.mkdir(parents=True)
        cfg = _make_tracking_config(
            wandb_enabled=False,
            index_path=str(tmp_path / "idx.json"),
        )
        t = Tracker(ctx, cfg)
        t.start({})
        t.finish()
        records = load_index(tmp_path / "idx.json")
        assert "wandb" not in records[0]


# ---------------------------------------------------------------------------
# E) TrackingConfig YAML loading
# ---------------------------------------------------------------------------


class TestTrackingConfigYaml:
    def test_tracking_defaults_when_section_absent(self, tmp_path):
        """YAML without a tracking: section should use TrackingConfig defaults."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            "dataset: toy\ndata_dir: data/toy_scale_easy\nmu: 1.0\n"
        )
        cfg = load_training_config(yaml_file)
        assert cfg.tracking.enabled is True
        assert cfg.tracking.wandb_enabled is False
        assert cfg.tracking.wandb_project == "higgsML-uncertainty"
        assert cfg.tracking.index_path == "results/runs_index.json"

    def test_tracking_section_overrides_defaults(self, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            "dataset: toy\ndata_dir: data/toy_scale_easy\nmu: 1.0\n"
            "tracking:\n"
            "  enabled: false\n"
            "  wandb_enabled: true\n"
            "  wandb_project: my-project\n"
            "  index_path: /tmp/idx.json\n"
        )
        cfg = load_training_config(yaml_file)
        assert cfg.tracking.enabled is False
        assert cfg.tracking.wandb_enabled is True
        assert cfg.tracking.wandb_project == "my-project"
        assert cfg.tracking.index_path == "/tmp/idx.json"

    def test_unknown_tracking_keys_are_ignored(self, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            "dataset: toy\ndata_dir: data/toy_scale_easy\nmu: 1.0\n"
            "tracking:\n"
            "  enabled: true\n"
            "  future_unknown_key: 999\n"
        )
        cfg = load_training_config(yaml_file)
        assert cfg.tracking.enabled is True
