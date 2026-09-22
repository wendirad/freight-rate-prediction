from types import SimpleNamespace
from unittest.mock import patch

import wandb


def test_run_sweep_trial_builds_config_and_delegates_to_run_tracked_experiment() -> None:
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
        }

        with patch(
            "experiments.sweep_agent.run_tracked_experiment",
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
