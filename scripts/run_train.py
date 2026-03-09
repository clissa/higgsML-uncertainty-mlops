#!/usr/bin/env python
"""Config-driven training entrypoint.

Usage::

    python scripts/run_train.py --config configs/train_toy.yaml
    python scripts/run_train.py --config configs/train_toy.yaml --seed 42
    python scripts/run_train.py --config configs/train_toy.yaml --output-dir /tmp/results
    python scripts/run_train.py --config configs/train_toy.yaml --run-name my-experiment

This replaces the older ``scripts/train.py`` hard-coded flow with a
YAML-driven pipeline that also records run metadata for reproducibility.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from conformal_predictions.config import TrainingConfig, load_training_config
from conformal_predictions.mlops.run_context import RunContext
from conformal_predictions.training.trainer import Trainer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the conformal-prediction training pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a training YAML config file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the seed from the config file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override the output directory from the config file.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Human-readable run label (defaults to run-<id>).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # ---- load config with CLI overrides ----
    cfg = load_training_config(args.config)

    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.run_name is not None:
        overrides["run_name"] = args.run_name
    if overrides:
        cfg = replace(cfg, **overrides)

    # ---- create run context ----
    ctx = RunContext.create(cfg, config_path=args.config)

    print(f"Run ID: {ctx.run_id}")
    print(f"Output: {ctx.output_dir}")
    print(f"Git commit: {ctx.git_commit or 'N/A'}")

    # ---- run training ----
    trainer = Trainer(cfg, ctx)
    trainer.run()

    print("\nDone.")


if __name__ == "__main__":
    main()
