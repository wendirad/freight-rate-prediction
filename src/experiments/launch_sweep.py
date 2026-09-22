"""Registers the lightgbm wandb sweep. Does not run any trials.

Launching the agent (wandb.agent(sweep_id, function=run_sweep_trial)) is a
separate, manual step. It must be done via the Python API, not the CLI
`wandb agent` command: configs/sweep_lightgbm.yaml defines no `program`/
`command` key, so the CLI agent has no script to exec and fails.
"""

from __future__ import annotations

import yaml

import wandb


def launch_sweep(
    prior_runs: list[str] | None = None, sweep_id: str | None = None
) -> str:
    """Registers a new sweep from configs/sweep_lightgbm.yaml, or resumes
    an existing one.

    sweep_id resumes an existing sweep instead of registering a new one:
    when given, no wandb.sweep() call is made at all and sweep_id is
    returned unchanged. wandb.agent can add more trials to any existing
    sweep just by being pointed at its ID, so resuming never needs (or
    should mint) a fresh sweep ID.

    prior_runs optionally warm-starts a newly created sweep's Bayesian
    search with an existing sweep's finished runs (run IDs, or
    entity/project/run_id paths), so the new search isn't starting cold.
    Ignored when sweep_id is given. Pass the config in memory as usual to
    narrow ranges from the CLI without editing configs/sweep_lightgbm.yaml.
    """
    if sweep_id is not None:
        return sweep_id

    with open("configs/sweep_lightgbm.yaml") as f:
        sweep_config = yaml.safe_load(f)

    with open("configs/training.yaml") as f:
        training_config = yaml.safe_load(f)

    wandb_project = training_config["wandb"]["project"]

    sweep_id = wandb.sweep(sweep_config, project=wandb_project, prior_runs=prior_runs)
    print(sweep_id)
    return sweep_id
