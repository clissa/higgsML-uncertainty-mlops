#!/usr/bin/env python
"""Config-driven pipeline entrypoint with selectable execution modes.

Usage::

    # Full pipeline (default)
    python scripts/run_train.py --config configs/train_toy.yaml

    # Train only
    python scripts/run_train.py --config configs/train_toy.yaml --mode train

    # Train + calibrate (no test evaluation)
    python scripts/run_train.py --config configs/train_toy.yaml --mode train+calibrate

    # All three stages
    python scripts/run_train.py --config configs/train_toy.yaml --mode all

CLI overrides (--seed, --output-dir, --run-name) are applied on top of
the YAML config.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from conformal_predictions.config import ModelConfig, load_training_config
from conformal_predictions.mlops.run_context import RunContext
from conformal_predictions.mlops.tracker import Tracker
from conformal_predictions.training.trainer import Trainer

VALID_MODES = ("train", "calibrate", "evaluate", "train+calibrate", "all")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the conformal-prediction pipeline.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a training YAML config file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=VALID_MODES,
        help=(
            "Execution mode.  'all' (default) runs train → calibrate → "
            "evaluate.  Individual stages can be selected."
        ),
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
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the model family. Currently only 'mlp' is supported.",
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
    if args.model is not None:
        overrides["model"] = ModelConfig(name=args.model)
    if overrides:
        cfg = replace(cfg, **overrides)

    # ---- create run context ----
    ctx = RunContext.create(cfg, config_path=args.config)

    print(f"Run ID: {ctx.run_id}")
    print(f"Output: {ctx.output_dir}")
    print(f"Model: {cfg.model.name}")
    print(f"Git commit: {ctx.git_commit or 'N/A'}")
    print(f"Mode: {args.mode}")

    # ---- start tracker ----
    tracker = Tracker(ctx, cfg.tracking)

    # ---- build trainer ----
    trainer = Trainer(cfg, ctx, tracker=tracker)

    # ---- data lineage (helper W&B runs — must precede tracker.start) ----
    trainer.prepare_data_lineage()

    # ---- start main W&B run ----
    tracker.start(cfg.to_dict())

    mode = args.mode

    if mode == "all":
        trainer.run()
    elif mode == "train":
        trainer.train()
        ctx.save_metadata()
    elif mode == "calibrate":
        trainer.train()
        trainer.calibrate()
        ctx.save_metadata()
    elif mode == "train+calibrate":
        trainer.train()
        trainer.calibrate()
        ctx.save_metadata()
    elif mode == "evaluate":
        trainer.train()
        cal = trainer.calibrate()
        trainer.evaluate(calibration_result=cal)
        ctx.save_metadata()

    tracker.finish()
    # Re-write the manifest so it includes metrics.json (registered by tracker.finish)
    ctx.save_manifest()

    print("\nDone.")


if __name__ == "__main__":
    main()
