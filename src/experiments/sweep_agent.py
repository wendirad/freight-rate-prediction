"""Sweep agent trial entry point.

Meant to be passed to wandb.agent(sweep_id, function=run_sweep_trial) from
the command line. Reads the sweep-assigned hyperparameters from
wandb.config and runs one tracked trial through run_tracked_experiment.
"""

from __future__ import annotations

import wandb
from experiments.run_experiment import DEFAULT_CACHE_DIR, load_config
from experiments.track_experiment import run_tracked_experiment

SWEEP_PARAM_KEYS = [
    "n_estimators",
    "learning_rate",
    "num_leaves",
    "max_depth",
    "subsample",
    "colsample_bytree",
    "random_state",
]


def run_sweep_trial() -> dict:
    wandb.init()

    config = load_config(configs_dir="configs", model_family="lightgbm")
    for key in SWEEP_PARAM_KEYS:
        if key in wandb.config:
            config["model"]["params"][key] = wandb.config[key]

    wandb_project = config["training"]["wandb"]["project"]

    return run_tracked_experiment(
        config=config,
        n_folds=5,
        cache_dir=DEFAULT_CACHE_DIR,
        wandb_project=wandb_project,
        wandb_run_name=f"sweep-trial-{wandb.run.id}",
        wandb_tags=["sweep", "lightgbm"],
    )
