"""Weights and Biases tracking wrapper around run_experiment.

Wraps the existing run_experiment call with wandb logging without
changing its internal logic, calling code, or return value.
"""

from __future__ import annotations

import wandb

from experiments.run_experiment import run_experiment


def run_tracked_experiment(
    config: dict,
    n_folds: int,
    cache_dir: str,
    wandb_project: str,
    wandb_run_name: str,
    wandb_tags: list[str],
) -> dict:
    run_already_active = wandb.run is not None

    if not run_already_active:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            tags=wandb_tags,
            config=config,
        )

    result = run_experiment(config, n_folds, cache_dir)

    wandb.log(
        {
            "mae_mean": result["mean"],
            "mae_std": result["std"],
            "mae_worst_fold": result["worst_fold"],
        }
    )

    fold_metrics_table = wandb.Table(
        columns=["fold_index", "mae", "rmse", "rmsle", "mape"]
    )
    for fold_index, fold_metric in enumerate(result["fold_metrics"]):
        fold_metrics_table.add_data(
            fold_index,
            fold_metric["mae"],
            fold_metric["rmse"],
            fold_metric["rmsle"],
            fold_metric["mape"],
        )
    wandb.log({"fold_metrics_table": fold_metrics_table})

    if not run_already_active:
        wandb.finish()

    return result
