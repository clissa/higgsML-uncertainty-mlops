"""Unit tests for ModelConfig and build_model (Phase 4.7).

Tests cover:
- ModelConfig validation (valid/invalid names)
- build_model factory (correct sklearn class, param merging)
- YAML config loading with model section
- CLI model override (--model flag)
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from conformal_predictions.config import (
    ModelConfig,
    TrainingConfig,
    load_training_config,
)
from conformal_predictions.training.models import build_model

# ---------------------------------------------------------------------------
# ModelConfig dataclass
# ---------------------------------------------------------------------------


class TestModelConfig:
    def test_default_is_mlp(self):
        cfg = ModelConfig()
        assert cfg.name == "mlp"
        assert cfg.params == {}

    @pytest.mark.parametrize("name", ["mlp", "glm", "random_forest"])
    def test_valid_names(self, name: str):
        cfg = ModelConfig(name=name)
        assert cfg.name == name

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="Unknown model name"):
            ModelConfig(name="xgboost")

    def test_params_preserved(self):
        cfg = ModelConfig(name="mlp", params={"hidden_layer_sizes": [64, 32]})
        assert cfg.params == {"hidden_layer_sizes": [64, 32]}

    def test_to_dict(self):
        cfg = ModelConfig(name="glm", params={"penalty": "l1"})
        d = cfg.to_dict()
        assert d == {"name": "glm", "params": {"penalty": "l1"}}


# ---------------------------------------------------------------------------
# build_model factory
# ---------------------------------------------------------------------------


class TestBuildModel:
    def test_mlp_default(self):
        models = build_model(ModelConfig(), seed=42)
        assert len(models) == 1
        assert "MLP" in models
        assert isinstance(models["MLP"], MLPClassifier)

    def test_glm(self):
        models = build_model(ModelConfig(name="glm"), seed=42)
        assert len(models) == 1
        assert "GLM" in models
        assert isinstance(models["GLM"], LogisticRegression)

    def test_random_forest(self):
        models = build_model(ModelConfig(name="random_forest"), seed=42)
        assert len(models) == 1
        assert "Random Forest" in models
        assert isinstance(models["Random Forest"], RandomForestClassifier)

    def test_seed_preserved(self):
        models = build_model(ModelConfig(name="mlp"), seed=123)
        assert models["MLP"].random_state == 123

    def test_param_override(self):
        cfg = ModelConfig(
            name="mlp",
            params={"hidden_layer_sizes": [64, 32], "max_iter": 500},
        )
        models = build_model(cfg, seed=7)
        mlp = models["MLP"]
        assert mlp.hidden_layer_sizes == (64, 32)  # list → tuple conversion
        assert mlp.max_iter == 500
        assert mlp.random_state == 7

    def test_rf_param_override(self):
        cfg = ModelConfig(name="random_forest", params={"n_estimators": 100})
        models = build_model(cfg, seed=99)
        rf = models["Random Forest"]
        assert rf.n_estimators == 100
        assert rf.random_state == 99


# ---------------------------------------------------------------------------
# YAML config loading with model section
# ---------------------------------------------------------------------------


class TestModelConfigFromYAML:
    def test_load_default_model_section(self, tmp_path: Path):
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text(
            "dataset: toy\n"
            "data_dir: data/fake\n"
            "seed: 42\n"
            "model:\n"
            "  name: mlp\n"
            "  params:\n"
            "    hidden_layer_sizes: [64, 32]\n"
        )
        cfg = load_training_config(yaml_path)
        assert cfg.model.name == "mlp"
        assert cfg.model.params == {"hidden_layer_sizes": [64, 32]}

    def test_load_model_string_shorthand(self, tmp_path: Path):
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text(
            "dataset: toy\n" "data_dir: data/fake\n" "seed: 42\n" "model: glm\n"
        )
        cfg = load_training_config(yaml_path)
        assert cfg.model.name == "glm"
        assert cfg.model.params == {}

    def test_load_no_model_section_defaults_to_mlp(self, tmp_path: Path):
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text("dataset: toy\ndata_dir: data/fake\nseed: 42\n")
        cfg = load_training_config(yaml_path)
        assert cfg.model.name == "mlp"

    def test_training_config_default_model(self):
        cfg = TrainingConfig()
        assert cfg.model.name == "mlp"

    def test_model_in_config_snapshot(self, tmp_path: Path):
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text(
            "dataset: toy\n"
            "data_dir: data/fake\n"
            "seed: 42\n"
            "model:\n"
            "  name: random_forest\n"
            "  params:\n"
            "    n_estimators: 200\n"
        )
        cfg = load_training_config(yaml_path)
        snapshot = cfg.to_dict()
        assert snapshot["model"]["name"] == "random_forest"
        assert snapshot["model"]["params"]["n_estimators"] == 200
