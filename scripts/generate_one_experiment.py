from __future__ import annotations

import argparse
from pathlib import Path

from conformal_predictions.data.toy import (
    generate_pseudo_experiment,
    load_toy_config_from_yaml,
    save_pseudo_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a single toy pseudo-experiment from a YAML config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config file (e.g. configs/toy_default.yaml).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="If provided, save the pseudo-experiment under this directory.",
    )
    parser.add_argument(
        "--id",
        type=str,
        default=None,
        help="Optional 16-char hex pseudo_experiment_id. If omitted, a random one is"
        " generated.",
    )
    args = parser.parse_args()

    cfg = load_toy_config_from_yaml(args.config)
    X, y, meta = generate_pseudo_experiment(cfg, pseudo_experiment_id=args.id)

    print("Generated pseudo-experiment")
    print(f" \tid: {meta['pseudo_experiment_id']}")
    print(f"\tmu_true: {meta['mu_true']}")
    print(f"\tgamma_true: {meta['gamma_true']}")
    print(f"\tbeta_true: {meta['beta_true']}")
    print("\nYields and sizes:")
    print(f"\tn_signal: {meta['n_signal']}")
    print(f"\tn_background: {meta['n_background']}")
    print(f"\tn_total: {meta['n_total']}")

    if args.outdir is not None:
        saved_path = save_pseudo_experiment(args.outdir, X, y, meta)
        print(f"Saved to: {saved_path}")

    print("---------------")


if __name__ == "__main__":
    main()
