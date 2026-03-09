"""Training configuration — YAML-loadable, frozen dataclass.

Unifies the ``Settings`` dataclasses that were previously hard-coded in
``scripts/train.py`` and ``scripts/train_higgs.py`` into a single,
config-driven schema.

Usage::

    from conformal_predictions.config import load_training_config
    cfg = load_training_config("configs/train_toy.yaml")
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional, Sequence

import yaml


@dataclass(frozen=True)
class TrainingConfig:
    """Frozen configuration for a training run.

    All fields have sensible defaults matching the toy pipeline so that
    ``TrainingConfig()`` is immediately usable for a quick local run.
    """

    # --- dataset identification ---
    dataset: str = "toy"  # "toy" or "higgs"
    data_dir: str = "data/toy_scale_easy"
    mu: float = 1.0

    # --- reproducibility ---
    seed: int = 18

    # --- data splitting (toy pipeline) ---
    test_prefixes: Sequence[str] = ("7e39", "6fcb")
    n_test_experiments: int = 1000
    valid_size: float = 0.2
    calib_size: float = 0.5

    # --- model / inference ---
    threshold: float = 0.5
    nonconf_target: str = "mu_hat"  # "mu_hat" or "n_pred"
    nonconf_method: str = "diff"  # "diff" or "abs"

    # --- output ---
    output_dir: str = "results"
    run_name: Optional[str] = None  # human-readable label; auto-generated if None

    def to_dict(self) -> dict:
        """Serialise to a plain dict (for JSON / run-context snapshots)."""
        d = asdict(self)
        # Convert Path-like strings for readability
        return d


def load_training_config(path: str | Path) -> TrainingConfig:
    """Load a ``TrainingConfig`` from a YAML file.

    Unknown keys in the YAML are silently ignored so that config files
    can carry extra metadata without breaking the loader (forward-compat).
    """
    path = Path(path)
    with open(path) as fh:
        raw: dict = yaml.safe_load(fh) or {}

    valid_keys = {f.name for f in fields(TrainingConfig)}
    filtered = {k: v for k, v in raw.items() if k in valid_keys}

    # Convert list → tuple for frozen dataclass compatibility
    if "test_prefixes" in filtered and isinstance(filtered["test_prefixes"], list):
        filtered["test_prefixes"] = tuple(filtered["test_prefixes"])

    return TrainingConfig(**filtered)
