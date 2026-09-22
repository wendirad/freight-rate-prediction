"""Weights and Biases tracking wrapper around run_experiment.

Wraps the existing run_experiment call with wandb logging without
changing its internal logic, calling code, or return value.
"""

from __future__ import annotations

import time

import wandb
from experiments.run_experiment import fit_final_model, run_experiment, run_loss_diagnostic


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


def run_tracked_loss_diagnostic(
    config: dict,
    n_folds: int,
    cache_dir: str,
    wandb_project: str,
    wandb_run_name: str,
    wandb_tags: list[str],
) -> dict:
    """Runs run_loss_diagnostic and logs all fold train/val MAE curves and
    best iterations to W&B, on top of the usual aggregate metrics.
    """
    run_already_active = wandb.run is not None

    if not run_already_active:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            tags=wandb_tags,
            config=config,
        )

    result = run_loss_diagnostic(config, n_folds, cache_dir)

    wandb.log(
        {
            "mae_mean": result["mean"],
            "mae_std": result["std"],
            "mae_worst_fold": result["worst_fold"],
        }
    )

    best_iteration_table = wandb.Table(
        columns=["fold_index", "best_iteration", "best_val_mae"]
    )
    for fold_index, diag in enumerate(result["fold_diagnostics"]):
        best_iteration_table.add_data(
            fold_index, diag["best_iteration"], diag["best_val_mae"]
        )

        iterations = list(range(1, len(diag["train_mae_curve"]) + 1))
        wandb.log(
            {
                f"fold_{fold_index}_loss_curve": wandb.plot.line_series(
                    xs=iterations,
                    ys=[diag["train_mae_curve"], diag["val_mae_curve"]],
                    keys=["train", "val"],
                    title=f"Fold {fold_index} train vs val MAE by boosting iteration",
                    xname="boosting_iteration",
                )
            }
        )

    wandb.log({"best_iterations": best_iteration_table})

    if not run_already_active:
        wandb.finish()

    return result


def run_tracked_diagnostic(
    config: dict,
    n_folds: int,
    cache_dir: str,
    wandb_project: str,
    wandb_run_name: str,
    wandb_tags: list[str],
) -> dict:
    """Like run_tracked_loss_diagnostic, but for a final-candidate training
    diagnostic: adds each fold's train/val MAE gap at its best iteration to
    the best-iterations table, and logs wall-clock runtime, dev-pool row
    count, and feature count alongside the usual aggregate metrics and
    per-fold curves. Used to decide overfitting/underfitting and a
    recommended final iteration count before any holdout evaluation.
    """
    run_already_active = wandb.run is not None

    if not run_already_active:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            tags=wandb_tags,
            config=config,
        )

    start = time.perf_counter()
    result = run_loss_diagnostic(config, n_folds, cache_dir)
    runtime_seconds = time.perf_counter() - start

    wandb.log(
        {
            "mae_mean": result["mean"],
            "mae_std": result["std"],
            "mae_worst_fold": result["worst_fold"],
            "runtime_seconds": runtime_seconds,
            "n_rows": result["n_rows"],
            "n_features": result["n_features"],
        }
    )

    best_iteration_table = wandb.Table(
        columns=["fold_index", "best_iteration", "best_val_mae", "mae_gap"]
    )
    for fold_index, diag in enumerate(result["fold_diagnostics"]):
        best_iteration_table.add_data(
            fold_index, diag["best_iteration"], diag["best_val_mae"], diag["mae_gap"]
        )

        iterations = list(range(1, len(diag["train_mae_curve"]) + 1))
        wandb.log(
            {
                f"fold_{fold_index}_loss_curve": wandb.plot.line_series(
                    xs=iterations,
                    ys=[diag["train_mae_curve"], diag["val_mae_curve"]],
                    keys=["train", "val"],
                    title=f"Fold {fold_index} train vs val MAE by boosting iteration",
                    xname="boosting_iteration",
                )
            }
        )

    wandb.log({"best_iterations": best_iteration_table})

    if not run_already_active:
        wandb.finish()

    result["runtime_seconds"] = runtime_seconds
    return result


def run_tracked_final_fit(
    config: dict,
    cache_dir: str,
    wandb_project: str,
    wandb_run_name: str,
    wandb_tags: list[str],
) -> dict:
    """Wraps fit_final_model with W&B tracking: logs the resolved config
    (automatic via wandb.init), the training curve, runtime, dev-pool row
    count, and feature count. This is the canonical record of a produced
    model artifact, so unlike the other tracked wrappers this always
    starts (and finishes) its own run rather than reusing an active one.
    """
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        tags=wandb_tags,
        config=config,
    )

    start = time.perf_counter()
    result = fit_final_model(config, cache_dir)
    runtime_seconds = time.perf_counter() - start

    wandb.log(
        {
            "in_sample_mae": result["in_sample_metrics"].get("mae"),
            "runtime_seconds": runtime_seconds,
            "n_rows": result["n_rows"],
            "n_features": result["n_features"],
        }
    )

    if result["train_curve"]:
        iterations = list(range(1, len(result["train_curve"]) + 1))
        wandb.log(
            {
                "training_curve": wandb.plot.line_series(
                    xs=iterations,
                    ys=[result["train_curve"]],
                    keys=["train"],
                    title="Final fit training MAE by boosting iteration",
                    xname="boosting_iteration",
                )
            }
        )

    wandb.finish()

    result["runtime_seconds"] = runtime_seconds
    return result


def log_train_val_loss_curve(
    model, key: str = "train_val_loss_curve", train_label: str = "training"
) -> None:
    """Logs a wandb line plot of train vs val loss per boosting iteration,
    read from a fitted LightGBM model's evals_result_ (requires the model
    to have been trained with Trainer.train(df, eval_df=...)).
    """
    evals_result = model.evals_result_
    val_label = next(name for name in evals_result if name != train_label)
    metric = next(iter(evals_result[train_label]))

    train_loss = evals_result[train_label][metric]
    val_loss = evals_result[val_label][metric]
    iterations = list(range(1, len(train_loss) + 1))

    wandb.log(
        {
            key: wandb.plot.line_series(
                xs=iterations,
                ys=[train_loss, val_loss],
                keys=["train", "val"],
                title=f"Train vs val {metric} by boosting iteration",
                xname="boosting_iteration",
            )
        }
    )
