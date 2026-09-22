"""Hydra-driven entry point for run_experiment.

Composes the config via Hydra (configs/config.yaml and its defaults list).
By default (workflow=final_fit) `uv run train` needs no overrides: it fits
one CatBoost model on the whole pre-October dev pool with the fixed
iteration count chosen from an earlier diagnostic run, saves a deployable
prediction bundle to artifacts/model.joblib. W&B is optional through
`tracker=wandb`. Diagnostics remain available via `workflow=diagnostic`
(5-fold CV, early stopping, per-fold curves) — there, W&B tracking is
opt-in and off by default (pass `tracker=wandb` to enable it; with it off,
run_loss_diagnostic is called directly and its own curve summary is
printed instead of anything going to wandb). `workflow=none` falls back
to a plain run_experiment CV pass, also with opt-in tracking.

No path here ever touches October (evaluate_holdout is never called).
Only workflow=final_fit saves an artifact; the other two never do.
"""

from __future__ import annotations

import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from experiments.run_experiment import (
    diagnostic_model_overrides,
    run_experiment,
    run_loss_diagnostic,
)

# An absolute path, not "../../configs": a relative config_path is resolved
# against the *importing* context, which differs between `python -m
# experiments.hydra_entry` and the installed `train` console script (the
# latter imports this module rather than executing it as __main__), and
# only the file-relative absolute form works reliably for both.
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent.parent / "configs")

_SPARK_BLOCKS = "▁▂▃▄▅▆▇█"


def _sparkline(values: list[float], width: int = 40) -> str:
    """Renders a compact unicode sparkline of a training curve, so the
    diagnostic report shows the actual shape of the curve instead of the
    underlying library's raw per-iteration/per-bin training log.
    """
    if not values:
        return ""
    lo, hi = min(values), max(values)
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values
    if hi == lo:
        return _SPARK_BLOCKS[0] * len(sampled)
    return "".join(
        _SPARK_BLOCKS[int((v - lo) / (hi - lo) * (len(_SPARK_BLOCKS) - 1))]
        for v in sampled
    )


def _summarize_diagnostic(result: dict) -> str:
    """Builds a plain-text overfitting/underfitting readout, a per-fold
    train/val MAE curve sparkline, and a recommended final iteration count
    from a run_loss_diagnostic (or run_tracked_diagnostic) result.
    """
    diagnostics = result["fold_diagnostics"]
    best_iterations = [d["best_iteration"] for d in diagnostics if d["best_iteration"]]
    gaps = [d["mae_gap"] for d in diagnostics if d["mae_gap"] is not None]

    lines = [
        f"mean/std/worst-fold MAE: {result['mean']:.2f} / {result['std']:.2f} / {result['worst_fold']:.2f}",
        f"rows: {result['n_rows']}, features: {result['n_features']}, runtime: {result['runtime_seconds']:.1f}s",
    ]

    for fold_index, diag in enumerate(diagnostics):
        train_curve = diag.get("train_mae_curve") or []
        val_curve = diag.get("val_mae_curve") or []
        if not train_curve or not val_curve:
            continue
        lines.append(
            f"fold {fold_index} (best_iter={diag['best_iteration']}, "
            f"best_val_mae={diag['best_val_mae']:.2f}):"
        )
        lines.append(f"  train {_sparkline(train_curve)}  [{train_curve[0]:.1f} -> {train_curve[-1]:.1f}]")
        lines.append(f"  val   {_sparkline(val_curve)}  [{val_curve[0]:.1f} -> {val_curve[-1]:.1f}]")

    if not best_iterations or not gaps:
        lines.append("No per-fold curves captured; cannot assess overfitting.")
        return "\n".join(lines)

    avg_gap = sum(gaps) / len(gaps)
    gap_ratio = avg_gap / result["mean"] if result["mean"] else float("nan")
    lines.append(
        f"per-fold best_iteration: {best_iterations} (min={min(best_iterations)}, "
        f"max={max(best_iterations)}, median={sorted(best_iterations)[len(best_iterations) // 2]})"
    )
    lines.append(f"avg train/val MAE gap at best iteration: {avg_gap:.2f} ({gap_ratio:.0%} of mean MAE)")

    if gap_ratio > 0.35:
        lines.append(
            "Assessment: OVERFITTING — the gap between train and validation MAE at "
            "each fold's best iteration is large relative to overall MAE. Early "
            "stopping is already compensating; consider more regularization "
            "(lower depth, higher l2_leaf_reg) before trusting a fixed iteration count."
        )
    elif gap_ratio < 0.10:
        lines.append(
            "Assessment: UNDERFITTING risk is low and the train/val gap is small; "
            "the model is not memorizing the fold. Fine to move to a fixed-iteration "
            "final fit."
        )
    else:
        lines.append(
            "Assessment: moderate train/val gap — typical, expected regularization "
            "tension from early stopping. Not a red flag on its own."
        )

    spread = max(best_iterations) - min(best_iterations)
    if spread > min(best_iterations):
        lines.append(
            f"best_iteration varies widely across folds (spread={spread}, more than "
            "the smallest fold's own value) — later folds see more training data and "
            "tolerate more rounds, so a single fixed count is a compromise, not a law."
        )

    recommended = max(best_iterations)
    lines.append(
        f"Recommended final iteration count: {recommended} (max best_iteration across "
        "folds, so the final full-dev-pool fit isn't cut short relative to what the "
        "largest fold needed). Re-verify with early stopping enabled on the full fit "
        "rather than trusting this number blindly."
    )
    return "\n".join(lines)


def _summarize_final_fit(result: dict, family: str, iterations: int, artifact_path: str) -> str:
    """Concise plain-text report for a run_tracked_final_fit result."""
    lines = [
        (
            f"Fit {family} on {result['n_rows']} rows, "
            f"{result['n_features']} features, fixed iterations={iterations}."
        ),
        (
            f"in-sample MAE: {result['in_sample_metrics'].get('mae'):.2f} "
            "(fit on all dev rows, not a generalization estimate)"
        ),
        f"runtime: {result['runtime_seconds']:.1f}s",
    ]
    if result["train_curve"]:
        curve = result["train_curve"]
        lines.append(f"training curve: {_sparkline(curve)}  [{curve[0]:.1f} -> {curve[-1]:.1f}]")
    lines.append(f"saved model + fitted preprocessing pipelines to {artifact_path}")
    lines.append("October holdout was not accessed.")
    return "\n".join(lines)


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> dict:
    config = OmegaConf.to_container(cfg, resolve=True)
    n_folds = config.pop("n_folds")
    cache_dir = config.pop("cache_dir")
    workflow_cfg = config.pop("workflow", {"name": "none"})
    tracker = config.pop("tracker", "none")
    use_wandb = tracker == "wandb"

    if workflow_cfg.get("name") == "final_fit":
        family = config["model"]["family"]
        config["model"]["params"].update(diagnostic_model_overrides(family, workflow_cfg))

        from serving.inference import PredictionBundle

        if use_wandb:
            from experiments.track_experiment import run_tracked_final_fit

            result = run_tracked_final_fit(
                config=config,
                cache_dir=cache_dir,
                wandb_project=config["training"]["wandb"]["project"],
                wandb_run_name=f"final-fit-{family}",
                wandb_tags=["final-fit", family],
            )
        else:
            from experiments.run_experiment import fit_final_model

            start = time.perf_counter()
            result = fit_final_model(config, cache_dir)
            result["runtime_seconds"] = time.perf_counter() - start

        bundle = PredictionBundle(
            cleaner=result["cleaner"],
            feature_pipeline=result["feature_pipeline"],
            trainer=result["trainer"],
            input_columns=result["raw_columns"],
            categorical_options=result["categorical_options"],
        )
        artifact_path = "artifacts/model.joblib"
        bundle.save(artifact_path)

        print(_summarize_final_fit(result, family, workflow_cfg["iterations"], artifact_path))
        return result

    if workflow_cfg.get("name") == "diagnostic":
        family = config["model"]["family"]
        config["model"]["params"].update(diagnostic_model_overrides(family, workflow_cfg))
        config["training"]["early_stopping_rounds"] = workflow_cfg["early_stopping_rounds"]

        if use_wandb:
            from experiments.track_experiment import run_tracked_diagnostic

            wandb_project = config["training"]["wandb"]["project"]
            result = run_tracked_diagnostic(
                config=config,
                n_folds=n_folds,
                cache_dir=cache_dir,
                wandb_project=wandb_project,
                wandb_run_name=f"final-candidate-diagnostic-{family}",
                wandb_tags=["diagnostic", "final-candidate", family],
            )
        else:
            start = time.perf_counter()
            result = run_loss_diagnostic(config, n_folds, cache_dir)
            result["runtime_seconds"] = time.perf_counter() - start
        print(_summarize_diagnostic(result))
        return result

    if use_wandb:
        from experiments.track_experiment import run_tracked_experiment

        wandb_project = config["training"]["wandb"]["project"]
        family = config["model"]["family"]
        result = run_tracked_experiment(
            config=config,
            n_folds=n_folds,
            cache_dir=cache_dir,
            wandb_project=wandb_project,
            wandb_run_name=f"run-{family}",
            wandb_tags=["hydra-entry", family],
        )
    else:
        result = run_experiment(config, n_folds, cache_dir)
    print(result)
    return result


if __name__ == "__main__":
    main()
