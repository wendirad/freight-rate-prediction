from unittest.mock import patch

import wandb


def test_run_tracked_experiment_logs_and_returns_result() -> None:
    with (
        patch.object(wandb, "init") as mock_init,
        patch.object(wandb, "log") as mock_log,
        patch.object(wandb, "finish") as mock_finish,
        patch.object(wandb, "Table") as mock_table,
    ):
        from experiments.track_experiment import run_tracked_experiment

        fake_result = {
            "metric": "mae",
            "mean": 10.5,
            "std": 1.2,
            "worst_fold": 12.0,
            "fold_metrics": [
                {"mae": 9.0, "rmse": 11.0, "rmsle": 0.5, "mape": 8.0},
                {"mae": 12.0, "rmse": 14.0, "rmsle": 0.6, "mape": 9.0},
            ],
        }

        with patch(
            "experiments.track_experiment.run_experiment", return_value=fake_result
        ):
            result = run_tracked_experiment(
                config={"data": {}},
                n_folds=5,
                cache_dir="cache",
                wandb_project="freight-rate-prediction",
                wandb_run_name="test-run",
                wandb_tags=["baseline"],
            )

        mock_init.assert_called_once()
        assert mock_log.call_count == 2

        first_log_call_args = mock_log.call_args_list[0][0][0]
        assert first_log_call_args["mae_mean"] == fake_result["mean"]
        assert first_log_call_args["mae_std"] == fake_result["std"]
        assert first_log_call_args["mae_worst_fold"] == fake_result["worst_fold"]

        mock_finish.assert_called_once()
        assert result == fake_result


def test_run_tracked_experiment_skips_init_and_finish_when_run_already_active() -> None:
    with (
        patch.object(wandb, "init") as mock_init,
        patch.object(wandb, "log") as mock_log,
        patch.object(wandb, "finish") as mock_finish,
        patch.object(wandb, "Table") as mock_table,
        patch.object(wandb, "run", new=object()),
    ):
        from experiments.track_experiment import run_tracked_experiment

        fake_result = {
            "metric": "mae",
            "mean": 20.0,
            "std": 2.5,
            "worst_fold": 25.0,
            "fold_metrics": [
                {"mae": 18.0, "rmse": 20.0, "rmsle": 0.4, "mape": 7.0},
                {"mae": 22.0, "rmse": 24.0, "rmsle": 0.5, "mape": 8.0},
            ],
        }

        with patch(
            "experiments.track_experiment.run_experiment", return_value=fake_result
        ):
            result = run_tracked_experiment(
                config={"data": {}},
                n_folds=5,
                cache_dir="cache",
                wandb_project="freight-rate-prediction",
                wandb_run_name="sweep-trial-abc123",
                wandb_tags=["sweep", "lightgbm"],
            )

        mock_init.assert_not_called()
        mock_finish.assert_not_called()

        assert mock_log.call_count == 2
        first_log_call_args = mock_log.call_args_list[0][0][0]
        assert first_log_call_args["mae_mean"] == fake_result["mean"]
        assert first_log_call_args["mae_std"] == fake_result["std"]
        assert first_log_call_args["mae_worst_fold"] == fake_result["worst_fold"]

        assert result == fake_result
