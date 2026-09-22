"""Sweep agent trial entry point.

Meant to be passed to wandb.agent(sweep_id, function=run_sweep_trial) from
the command line. Reads the sweep-assigned hyperparameters from
wandb.config and runs one tracked trial through run_tracked_loss_diagnostic,
which trains each fold with early stopping (configs/training.yaml's
early_stopping_rounds) and logs mean/std/worst-fold MAE plus each fold's
best iteration.
"""

from __future__ import annotations

import wandb
from experiments.run_experiment import DEFAULT_CACHE_DIR, load_config
from experiments.track_experiment import (
    run_tracked_experiment,
    run_tracked_loss_diagnostic,
)

SWEEP_PARAM_KEYS = [
    "n_estimators",
    "learning_rate",
    "num_leaves",
    "max_depth",
    "subsample",
    "colsample_bytree",
]

CATBOOST_SWEEP_PARAM_KEYS = [
    "iterations",
    "learning_rate",
    "depth",
    "l2_leaf_reg",
]


def run_sweep_trial() -> dict:
    wandb.init()

    config = load_config(configs_dir="configs", model_family="lightgbm")
    for key in SWEEP_PARAM_KEYS:
        if key in wandb.config:
            config["model"]["params"][key] = wandb.config[key]

    wandb_project = config["training"]["wandb"]["project"]

    return run_tracked_loss_diagnostic(
        config=config,
        n_folds=5,
        cache_dir=DEFAULT_CACHE_DIR,
        wandb_project=wandb_project,
        wandb_run_name=f"sweep-trial-{wandb.run.id}",
        wandb_tags=["sweep", "lightgbm"],
    )


def run_catboost_sweep_trial() -> dict:
    """Like run_sweep_trial, but for the CatBoost baseline
    (configs/sweep_catboost.yaml). Uses run_tracked_experiment, not the
    loss-diagnostic path: CatBoost's early-stopping mechanism isn't
    callback-compatible with Trainer.train's lgb.early_stopping wiring, so
    this sweep doesn't early-stop, matching the CatBoost default baseline
    that was run without it.
    """
    wandb.init()

    config = load_config(configs_dir="configs", model_family="catboost")
    for key in CATBOOST_SWEEP_PARAM_KEYS:
        if key in wandb.config:
            config["model"]["params"][key] = wandb.config[key]

    wandb_project = config["training"]["wandb"]["project"]

    return run_tracked_experiment(
        config=config,
        n_folds=5,
        cache_dir=DEFAULT_CACHE_DIR,
        wandb_project=wandb_project,
        wandb_run_name=f"sweep-trial-{wandb.run.id}",
        wandb_tags=["sweep", "catboost"],
    )
