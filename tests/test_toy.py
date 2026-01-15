import numpy as np

from conformal_predictions.data.toy import (
    ToyConfig,
    generate_pseudo_experiment,
    generate_pseudo_experiment_from_yaml,
)


def test_generation():
    cfg = ToyConfig(
        mu=1.0,
        gamma=336.0,
        beta=664.0,
        n_features=2,
        signal_mean=np.array([1.0, 1.0]),
        signal_std=np.array([1.0, 1.0]),
        signal_rho=np.array([0.2]),
        background_mean=np.array([0.0, 0.0]),
        background_std=np.array([1.0, 1.0]),
        background_rho=np.array([0.0]),
        signal_weight=336.0,
        background_weight=664.0,
        seed=123,
    )

    X, y, meta = generate_pseudo_experiment(cfg)

    # basic shape checks
    assert X.ndim == 2
    assert X.shape[1] == cfg.n_features
    assert y.shape[0] == X.shape[0]
    assert meta["weights"].shape[0] == X.shape[0]

    # label sanity
    assert set(np.unique(y)).issubset({0, 1})

    # counts consistency
    n_signal = int((y == 1).sum())
    n_background = int((y == 0).sum())
    assert n_signal == meta["n_signal"]
    assert n_background == meta["n_background"]
    assert n_signal + n_background == meta["n_total"]

    # yield-conserving weights (only check if non-empty)
    if n_signal > 0:
        assert np.isclose(
            meta["weights"][y == 1].sum(),
            cfg.signal_weight,
            rtol=1e-6,
        )
    if n_background > 0:
        assert np.isclose(
            meta["weights"][y == 0].sum(),
            cfg.background_weight,
            rtol=1e-6,
        )

    # metadata sanity
    assert meta["mu_true"] == cfg.mu
    assert meta["gamma_true"] == cfg.gamma
    assert meta["beta_true"] == cfg.beta
    assert "feature_params" in meta


def test_generation_from_yaml(tmp_path):
    # Create a minimal YAML config file
    yaml_content = """
mu: 1.0
gamma: 336.0
beta: 664.0
n_features: 2

signal_mean: [1.0, 1.0]
signal_std: [1.0, 1.0]
signal_rho: [0.2]

background_mean: [0.0, 0.0]
background_std: [1.0, 1.0]
background_rho: [0.0]

signal_weight: 336.0
background_weight: 664.0
seed: 123
"""
    config_path = tmp_path / "toy_config.yaml"
    config_path.write_text(yaml_content)

    # Generate pseudo-experiment from YAML
    X, y, meta = generate_pseudo_experiment_from_yaml(
        yaml_path=config_path,
        pseudo_experiment_id="abcdef0123456789",
    )

    # Basic sanity checks
    assert X.ndim == 2
    assert X.shape[1] == 2
    assert y.shape[0] == X.shape[0]
    assert meta["weights"].shape[0] == X.shape[0]

    # Labels sanity
    assert set(np.unique(y)).issubset({0, 1})

    # Metadata consistency
    assert meta["mu_true"] == 1.0
    assert meta["gamma_true"] == 336.0
    assert meta["beta_true"] == 664.0
    assert meta["pseudo_experiment_id"] == "abcdef0123456789"

    # Yield-conserving weights (only if non-empty)
    if meta["n_signal"] > 0:
        assert np.isclose(
            meta["weights"][y == 1].sum(),
            336.0,
            rtol=1e-6,
        )
    if meta["n_background"] > 0:
        assert np.isclose(
            meta["weights"][y == 0].sum(),
            664.0,
            rtol=1e-6,
        )


def test_reproducibility():
    cfg = ToyConfig(
        mu=1.0,
        gamma=336.0,
        beta=664.0,
        n_features=2,
        signal_mean=np.array([1.0, 1.0]),
        signal_std=np.array([1.0, 1.0]),
        signal_rho=np.array([0.2]),
        background_mean=np.array([0.0, 0.0]),
        background_std=np.array([1.0, 1.0]),
        background_rho=np.array([0.0]),
        signal_weight=336.0,
        background_weight=664.0,
        seed=123,
    )

    pseudo_experiment_id = "0123456789abcdef"  # 16-char hex string

    X1, y1, meta1 = generate_pseudo_experiment(
        cfg, pseudo_experiment_id=pseudo_experiment_id
    )
    X2, y2, meta2 = generate_pseudo_experiment(
        cfg, pseudo_experiment_id=pseudo_experiment_id
    )

    assert meta1["pseudo_experiment_id"] == pseudo_experiment_id
    assert meta2["pseudo_experiment_id"] == pseudo_experiment_id

    # Exact equality is expected for fixed seed + fixed ID
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)
    assert np.array_equal(meta1["weights"], meta2["weights"])

    # Optional: counts match (redundant but nice)
    assert meta1["n_total"] == meta2["n_total"]
    assert meta1["n_signal"] == meta2["n_signal"]
    assert meta1["n_background"] == meta2["n_background"]
