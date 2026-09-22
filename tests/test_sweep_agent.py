from types import SimpleNamespace
from unittest.mock import patch

import wandb
from experiments.sweep_agent import SWEEP_PARAM_KEYS


def test_random_state_is_not_a_sweep_controlled_parameter() -> None:
    assert "random_state" not in SWEEP_PARAM_KEYS


def test_run_sweep_trial_builds_config_and_delegates_to_run_tracked_loss_diagnostic() -> None:
    with (
        patch.object(wandb, "init") as mock_init,
        patch.object(
            wandb, "config", new={"n_estimators": 250, "learning_rate": 0.1}
        ),
        patch.object(wandb, "run", new=SimpleNamespace(id="abc123")),
    ):
        from experiments.sweep_agent import run_sweep_trial

        fake_result = {
            "metric": "mae",
            "mean": 15.0,
            "std": 3.0,
            "worst_fold": 18.0,
            "fold_metrics": [{"mae": 15.0, "rmse": 17.0, "rmsle": 0.4, "mape": 6.0}],
            "fold_diagnostics": [
                {
                    "overall": {"mae": 15.0, "rmse": 17.0, "rmsle": 0.4, "mape": 6.0},
                    "train_mae_curve": [20.0, 15.0],
                    "val_mae_curve": [22.0, 15.0],
                    "best_iteration": 2,
                    "best_val_mae": 15.0,
                }
            ],
        }

        with patch(
            "experiments.sweep_agent.run_tracked_loss_diagnostic",
            return_value=fake_result,
        ) as mock_run_tracked:
            result = run_sweep_trial()

        mock_init.assert_called_once_with()

        assert mock_run_tracked.call_count == 1
        call_kwargs = mock_run_tracked.call_args.kwargs

        assert call_kwargs["config"]["model"]["params"]["n_estimators"] == 250
        assert call_kwargs["config"]["model"]["params"]["learning_rate"] == 0.1
        assert call_kwargs["config"]["model"]["family"] == "lightgbm"
        assert call_kwargs["n_folds"] == 5
        assert call_kwargs["cache_dir"] == "data/interim/cache"
        assert call_kwargs["wandb_project"] == "freight-rate-prediction"
        assert call_kwargs["wandb_run_name"] == "sweep-trial-abc123"
        assert call_kwargs["wandb_tags"] == ["sweep", "lightgbm"]

        assert result == fake_result


def test_stray_random_state_in_wandb_config_does_not_override_fixed_seed() -> None:
    """A sweep sweeping over other hyperparameters must never let a stray
    random_state key in wandb.config (e.g. from an old sweep definition or
    manual override) overwrite the fixed seed in configs/model/lightgbm.yaml.
    """
    with (
        patch.object(wandb, "init"),
        patch.object(
            wandb,
            "config",
            new={"n_estimators": 300, "random_state": 999},
        ),
        patch.object(wandb, "run", new=SimpleNamespace(id="xyz789")),
    ):
        from experiments.sweep_agent import run_sweep_trial

        fake_result = {
            "metric": "mae",
            "mean": 10.0,
            "std": 1.0,
            "worst_fold": 11.0,
            "fold_metrics": [],
            "fold_diagnostics": [],
        }

        with patch(
            "experiments.sweep_agent.run_tracked_loss_diagnostic",
            return_value=fake_result,
        ) as mock_run_tracked:
            run_sweep_trial()

        call_kwargs = mock_run_tracked.call_args.kwargs
        assert call_kwargs["config"]["model"]["params"]["n_estimators"] == 300
        assert call_kwargs["config"]["model"]["params"]["random_state"] == 42


def test_run_catboost_sweep_trial_builds_config_and_delegates_to_run_tracked_experiment() -> None:
    with (
        patch.object(wandb, "init") as mock_init,
        patch.object(wandb, "config", new={"iterations": 300, "depth": 4}),
        patch.object(wandb, "run", new=SimpleNamespace(id="cb123")),
    ):
        from experiments.sweep_agent import run_catboost_sweep_trial

        fake_result = {
            "metric": "mae",
            "mean": 120.0,
            "std": 10.0,
            "worst_fold": 130.0,
            "fold_metrics": [],
        }

        with patch(
            "experiments.sweep_agent.run_tracked_experiment",
            return_value=fake_result,
        ) as mock_run_tracked:
            result = run_catboost_sweep_trial()

        mock_init.assert_called_once_with()

        call_kwargs = mock_run_tracked.call_args.kwargs
        assert call_kwargs["config"]["model"]["params"]["iterations"] == 300
        assert call_kwargs["config"]["model"]["params"]["depth"] == 4
        assert call_kwargs["config"]["model"]["family"] == "catboost"
        assert call_kwargs["n_folds"] == 5
        assert call_kwargs["wandb_run_name"] == "sweep-trial-cb123"
        assert call_kwargs["wandb_tags"] == ["sweep", "catboost"]

        assert result == fake_result
