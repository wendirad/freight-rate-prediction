"""Registers the lightgbm wandb sweep. Does not run any trials.

Launching the agent (wandb.agent(sweep_id, function=run_sweep_trial)) is a
separate, manual command-line step.
"""

from __future__ import annotations

import yaml
import wandb


def launch_sweep() -> str:
    with open("configs/sweep_lightgbm.yaml") as f:
        sweep_config = yaml.safe_load(f)

    with open("configs/training.yaml") as f:
        training_config = yaml.safe_load(f)

    wandb_project = training_config["wandb"]["project"]

    sweep_id = wandb.sweep(sweep_config, project=wandb_project)
    print(sweep_id)
    return sweep_id
