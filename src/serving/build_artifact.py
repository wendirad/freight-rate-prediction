"""Command-line entry point for fitting a deployable prediction bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from serving.inference import build_prediction_bundle

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/model.joblib")
    parser.add_argument(
        "--include-holdout",
        action="store_true",
        help="Fit on all labeled rows after final holdout evaluation.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides such as model=lightgbm experiment=baseline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        composed = compose(config_name="config", overrides=args.overrides)

    config = OmegaConf.to_container(composed, resolve=True)
    cache_dir = config.pop("cache_dir")
    config.pop("n_folds", None)
    config.pop("workflow", None)

    bundle = build_prediction_bundle(
        config,
        cache_dir=cache_dir,
        include_holdout=args.include_holdout,
    )
    bundle.save(args.output)
    print(f"Saved prediction bundle to {args.output}")


if __name__ == "__main__":
    main()
