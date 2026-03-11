"""Tests for Phase 4.9-A W&B artifact helpers.

Covers:
- artifact_name() naming convention
- log_or_use_data_artifact() with "latest" → log_artifact called
- log_or_use_data_artifact() with pinned version → use_artifact called
- No-op when wandb_run is None
- Split metadata included in artifact
- log_model_artifact() uses add_file
- artifact_version parsed from YAML config
- Tracker convenience methods delegate correctly
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from conformal_predictions.config import TrackingConfig, load_training_config
from conformal_predictions.mlops.artifacts import (
    artifact_name,
    log_model_artifact,
    log_or_use_data_artifact,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_wandb_run() -> MagicMock:
    """Return a mock that behaves like a wandb.Run."""
    run = MagicMock()
    run.log_artifact = MagicMock()
    run.use_artifact = MagicMock()
    return run


# ---------------------------------------------------------------------------
# A) artifact_name
# ---------------------------------------------------------------------------


class TestArtifactName:
    def test_basic(self):
        assert artifact_name("toy", 1.0, "train") == "toy-1.0-train"

    def test_different_splits(self):
        assert artifact_name("toy", 1.0, "raw") == "toy-1.0-raw"
        assert artifact_name("toy", 1.0, "val") == "toy-1.0-val"
        assert artifact_name("toy", 1.0, "calib") == "toy-1.0-calib"
        assert artifact_name("toy", 1.0, "test") == "toy-1.0-test"

    def test_different_dataset_and_mu(self):
        assert artifact_name("higgs", 0.5, "train") == "higgs-0.5-train"

    def test_mu_formatting(self):
        # mu is a float; the f-string renders it as-is
        assert artifact_name("toy", 0.0, "raw") == "toy-0.0-raw"


# ---------------------------------------------------------------------------
# B) log_or_use_data_artifact — latest (log new)
# ---------------------------------------------------------------------------


class TestLogOrUseDataArtifactLatest:
    """version="latest" should create + log a new artifact."""

    @patch("conformal_predictions.mlops.artifacts._WANDB_AVAILABLE", True)
    @patch("conformal_predictions.mlops.artifacts._wandb")
    def test_log_artifact_called(self, mock_wandb, tmp_path):
        run = _mock_wandb_run()
        fake_art = MagicMock()
        fake_art.metadata = {}
        mock_wandb.Artifact.return_value = fake_art

        f1 = tmp_path / "data.npz"
        f1.touch()

        result = log_or_use_data_artifact(run, "toy-1.0-train", [f1], version="latest")

        mock_wandb.Artifact.assert_called_once_with("toy-1.0-train", type="dataset")
        fake_art.add_reference.assert_called_once_with(f"file://{f1.resolve()}")
        run.log_artifact.assert_called_once_with(fake_art)
        assert result is fake_art

    @patch("conformal_predictions.mlops.artifacts._WANDB_AVAILABLE", True)
    @patch("conformal_predictions.mlops.artifacts._wandb")
    def test_multiple_files_referenced(self, mock_wandb, tmp_path):
        run = _mock_wandb_run()
        fake_art = MagicMock()
        fake_art.metadata = {}
        mock_wandb.Artifact.return_value = fake_art

        files = [tmp_path / f"file{i}.npz" for i in range(3)]
        for f in files:
            f.touch()

        log_or_use_data_artifact(run, "toy-1.0-raw", files, version="latest")

        assert fake_art.add_reference.call_count == 3
        for f in files:
            fake_art.add_reference.assert_any_call(f"file://{f.resolve()}")

    @patch("conformal_predictions.mlops.artifacts._WANDB_AVAILABLE", True)
    @patch("conformal_predictions.mlops.artifacts._wandb")
    def test_split_metadata_included(self, mock_wandb, tmp_path):
        run = _mock_wandb_run()
        fake_art = MagicMock()
        fake_art.metadata = {}
        mock_wandb.Artifact.return_value = fake_art

        f1 = tmp_path / "data.npz"
        f1.touch()
        params = {"seed": 42, "valid_size": 100, "calib_size": 200}

        log_or_use_data_artifact(
            run, "toy-1.0-train", [f1], split_params=params, version="latest"
        )

        assert fake_art.metadata == params

    @patch("conformal_predictions.mlops.artifacts._WANDB_AVAILABLE", True)
    @patch("conformal_predictions.mlops.artifacts._wandb")
    def test_no_metadata_when_none(self, mock_wandb, tmp_path):
        run = _mock_wandb_run()
        fake_art = MagicMock()
        fake_art.metadata = {}
        mock_wandb.Artifact.return_value = fake_art

        f1 = tmp_path / "data.npz"
        f1.touch()

        log_or_use_data_artifact(run, "toy-1.0-train", [f1], version="latest")

        # metadata dict was never updated
        assert fake_art.metadata == {}


# ---------------------------------------------------------------------------
# C) log_or_use_data_artifact — pinned version
# ---------------------------------------------------------------------------


class TestLogOrUseDataArtifactPinned:
    """A pinned version (e.g. "v3") should call use_artifact, not log."""

    @patch("conformal_predictions.mlops.artifacts._WANDB_AVAILABLE", True)
    @patch("conformal_predictions.mlops.artifacts._wandb")
    def test_use_artifact_called(self, mock_wandb, tmp_path):
        run = _mock_wandb_run()
        expected = MagicMock()
        run.use_artifact.return_value = expected

        result = log_or_use_data_artifact(run, "toy-1.0-train", [], version="v3")

        run.use_artifact.assert_called_once_with("toy-1.0-train:v3")
        run.log_artifact.assert_not_called()
        assert result is expected

    @patch("conformal_predictions.mlops.artifacts._WANDB_AVAILABLE", True)
    @patch("conformal_predictions.mlops.artifacts._wandb")
    def test_pinned_v0(self, mock_wandb):
        run = _mock_wandb_run()
        log_or_use_data_artifact(run, "toy-1.0-raw", [], version="v0")
        run.use_artifact.assert_called_once_with("toy-1.0-raw:v0")


# ---------------------------------------------------------------------------
# D) No-op when wandb_run is None
# ---------------------------------------------------------------------------


class TestNoOpWhenNone:
    def test_data_artifact_noop(self, tmp_path):
        result = log_or_use_data_artifact(None, "toy-1.0-train", [], version="latest")
        assert result is None

    def test_data_artifact_pinned_noop(self):
        result = log_or_use_data_artifact(None, "toy-1.0-train", [], version="v3")
        assert result is None

    def test_model_artifact_noop(self, tmp_path):
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        result = log_model_artifact(None, model_dir, "mlp")
        assert result is None

    @patch("conformal_predictions.mlops.artifacts._WANDB_AVAILABLE", False)
    def test_data_artifact_noop_wandb_unavailable(self):
        run = _mock_wandb_run()
        result = log_or_use_data_artifact(run, "toy-1.0-train", [], version="latest")
        assert result is None
        run.log_artifact.assert_not_called()

    @patch("conformal_predictions.mlops.artifacts._WANDB_AVAILABLE", False)
    def test_model_artifact_noop_wandb_unavailable(self, tmp_path):
        run = _mock_wandb_run()
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        result = log_model_artifact(run, model_dir, "mlp")
        assert result is None
        run.log_artifact.assert_not_called()


# ---------------------------------------------------------------------------
# E) log_model_artifact — add_file
# ---------------------------------------------------------------------------


class TestLogModelArtifact:
    @patch("conformal_predictions.mlops.artifacts._WANDB_AVAILABLE", True)
    @patch("conformal_predictions.mlops.artifacts._wandb")
    def test_model_files_uploaded(self, mock_wandb, tmp_path):
        run = _mock_wandb_run()
        fake_art = MagicMock()
        mock_wandb.Artifact.return_value = fake_art

        model_dir = tmp_path / "models"
        model_dir.mkdir()
        (model_dir / "mlp.joblib").write_bytes(b"model-data")
        (model_dir / "scaler.joblib").write_bytes(b"scaler-data")

        result = log_model_artifact(run, model_dir, "mlp")

        mock_wandb.Artifact.assert_called_once_with("mlp", type="model")
        assert fake_art.add_file.call_count == 2
        run.log_artifact.assert_called_once_with(fake_art)
        assert result is fake_art

    @patch("conformal_predictions.mlops.artifacts._WANDB_AVAILABLE", True)
    @patch("conformal_predictions.mlops.artifacts._wandb")
    def test_model_artifact_name_format(self, mock_wandb, tmp_path):
        run = _mock_wandb_run()
        fake_art = MagicMock()
        mock_wandb.Artifact.return_value = fake_art

        model_dir = tmp_path / "models"
        model_dir.mkdir()
        (model_dir / "bdt.joblib").write_bytes(b"data")

        log_model_artifact(run, model_dir, "bdt")

        mock_wandb.Artifact.assert_called_once_with("bdt", type="model")

    @patch("conformal_predictions.mlops.artifacts._WANDB_AVAILABLE", True)
    @patch("conformal_predictions.mlops.artifacts._wandb")
    def test_empty_model_dir(self, mock_wandb, tmp_path):
        """An empty model dir should still log the artifact (0 files)."""
        run = _mock_wandb_run()
        fake_art = MagicMock()
        mock_wandb.Artifact.return_value = fake_art

        model_dir = tmp_path / "models"
        model_dir.mkdir()

        result = log_model_artifact(run, model_dir, "mlp")

        fake_art.add_file.assert_not_called()
        run.log_artifact.assert_called_once_with(fake_art)
        assert result is fake_art

    @patch("conformal_predictions.mlops.artifacts._WANDB_AVAILABLE", True)
    @patch("conformal_predictions.mlops.artifacts._wandb")
    def test_skips_subdirectories(self, mock_wandb, tmp_path):
        """Only files (not subdirectories) should be added."""
        run = _mock_wandb_run()
        fake_art = MagicMock()
        mock_wandb.Artifact.return_value = fake_art

        model_dir = tmp_path / "models"
        model_dir.mkdir()
        (model_dir / "mlp.joblib").write_bytes(b"data")
        (model_dir / "subdir").mkdir()

        log_model_artifact(run, model_dir, "mlp")

        assert fake_art.add_file.call_count == 1


# ---------------------------------------------------------------------------
# F) artifact_version parsed from YAML
# ---------------------------------------------------------------------------


class TestArtifactVersionConfig:
    def test_default_artifact_version(self):
        cfg = TrackingConfig()
        assert cfg.artifact_version == "latest"

    def test_custom_artifact_version(self):
        cfg = TrackingConfig(artifact_version="v3")
        assert cfg.artifact_version == "v3"

    def test_artifact_version_from_yaml(self, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            "dataset: toy\n"
            "data_dir: data/toy\n"
            "mu: 1.0\n"
            "tracking:\n"
            "  artifact_version: v5\n"
        )
        cfg = load_training_config(yaml_file)
        assert cfg.tracking.artifact_version == "v5"

    def test_artifact_version_defaults_when_absent(self, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            "dataset: toy\n"
            "data_dir: data/toy\n"
            "mu: 1.0\n"
            "tracking:\n"
            "  wandb_enabled: false\n"
        )
        cfg = load_training_config(yaml_file)
        assert cfg.tracking.artifact_version == "latest"


# ---------------------------------------------------------------------------
# G) Tracker convenience methods
# ---------------------------------------------------------------------------


class TestTrackerArtifactMethods:
    """Verify Tracker delegates to artifacts.py correctly."""

    def _make_tracker(self, artifact_version: str = "latest"):
        from conformal_predictions.mlops.tracker import Tracker

        ctx = MagicMock()
        ctx.output_dir = Path("/tmp/fake-run")
        cfg = TrackingConfig(
            enabled=True,
            wandb_enabled=False,
            artifact_version=artifact_version,
        )
        return Tracker(ctx, cfg)

    def test_log_data_artifact_delegates(self, tmp_path):
        tracker = self._make_tracker()
        f1 = tmp_path / "data.npz"
        f1.touch()

        with patch(
            "conformal_predictions.mlops.artifacts.log_or_use_data_artifact"
        ) as mock_log:
            mock_log.return_value = None
            tracker.log_data_artifact("toy-1.0-train", [f1], split_params={"seed": 1})
            mock_log.assert_called_once_with(
                tracker._wandb_run,
                "toy-1.0-train",
                [f1],
                split_params={"seed": 1},
                version="latest",
            )

    def test_log_data_artifact_uses_config_version(self, tmp_path):
        tracker = self._make_tracker(artifact_version="v2")
        f1 = tmp_path / "data.npz"
        f1.touch()

        with patch(
            "conformal_predictions.mlops.artifacts.log_or_use_data_artifact"
        ) as mock_log:
            mock_log.return_value = None
            tracker.log_data_artifact("toy-1.0-train", [f1])
            mock_log.assert_called_once_with(
                tracker._wandb_run,
                "toy-1.0-train",
                [f1],
                split_params=None,
                version="v2",
            )

    def test_log_model_artifact_delegates(self, tmp_path):
        tracker = self._make_tracker()

        with patch(
            "conformal_predictions.mlops.artifacts.log_model_artifact"
        ) as mock_log:
            mock_log.return_value = None
            tracker.log_model_artifact(tmp_path / "models", "mlp")
            mock_log.assert_called_once_with(
                tracker._wandb_run,
                tmp_path / "models",
                "mlp",
            )

    def test_use_data_artifact_noop_no_wandb(self):
        tracker = self._make_tracker()
        # _wandb_run is None because wandb_enabled=False
        result = tracker.use_data_artifact("toy-1.0-calib")
        assert result is None
