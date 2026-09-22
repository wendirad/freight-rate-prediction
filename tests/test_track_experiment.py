from unittest.mock import patch

import wandb


def test_run_tracked_experiment_logs_and_returns_result() -> None:
    with (
        patch.object(wandb, "init") as mock_init,
        patch.object(wandb, "log") as mock_log,
        patch.object(wandb, "finish") as mock_finish,
        patch.object(wandb, "Table") as _mock_table,
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


def test_run_tracked_loss_diagnostic_logs_fold_curves_and_best_iterations() -> None:
    with (
        patch.object(wandb, "init") as mock_init,
        patch.object(wandb, "log") as mock_log,
        patch.object(wandb, "finish") as mock_finish,
        patch.object(wandb, "Table") as _mock_table,
        patch.object(wandb, "plot") as _mock_plot,
    ):
        from experiments.track_experiment import run_tracked_loss_diagnostic

        fake_result = {
            "metric": "mae",
            "mean": 10.0,
            "std": 1.0,
            "worst_fold": 11.0,
            "fold_metrics": [{"mae": 9.0}, {"mae": 11.0}],
            "fold_diagnostics": [
                {
                    "overall": {"mae": 9.0},
                    "train_mae_curve": [20.0, 15.0, 10.0],
                    "val_mae_curve": [22.0, 18.0, 9.0],
                    "best_iteration": 3,
                    "best_val_mae": 9.0,
                },
                {
                    "overall": {"mae": 11.0},
                    "train_mae_curve": [21.0, 16.0],
                    "val_mae_curve": [23.0, 11.0],
                    "best_iteration": 2,
                    "best_val_mae": 11.0,
                },
            ],
        }

        with patch(
            "experiments.track_experiment.run_loss_diagnostic",
            return_value=fake_result,
        ):
            result = run_tracked_loss_diagnostic(
                config={"data": {}},
                n_folds=2,
                cache_dir="cache",
                wandb_project="freight-rate-prediction",
                wandb_run_name="loss-diagnostic",
                wandb_tags=["loss-diagnostic"],
            )

        mock_init.assert_called_once()
        # aggregate metrics + one loss-curve log per fold + best-iteration table
        assert mock_log.call_count == 1 + len(fake_result["fold_diagnostics"]) + 1

        aggregate_call = mock_log.call_args_list[0][0][0]
        assert aggregate_call["mae_mean"] == fake_result["mean"]

        assert _mock_table.call_count == 1
        assert _mock_table.return_value.add_data.call_count == 2
        _mock_table.return_value.add_data.assert_any_call(0, 3, 9.0)
        _mock_table.return_value.add_data.assert_any_call(1, 2, 11.0)

        mock_finish.assert_called_once()
        assert result == fake_result


def test_run_tracked_diagnostic_logs_gap_runtime_rows_and_features() -> None:
    with (
        patch.object(wandb, "init") as mock_init,
        patch.object(wandb, "log") as mock_log,
        patch.object(wandb, "finish") as mock_finish,
        patch.object(wandb, "Table") as _mock_table,
        patch.object(wandb, "plot") as _mock_plot,
    ):
        from experiments.track_experiment import run_tracked_diagnostic

        fake_result = {
            "metric": "mae",
            "mean": 10.0,
            "std": 1.0,
            "worst_fold": 11.0,
            "n_rows": 1000,
            "n_features": 14,
            "fold_metrics": [{"mae": 9.0}, {"mae": 11.0}],
            "fold_diagnostics": [
                {
                    "overall": {"mae": 9.0},
                    "train_mae_curve": [20.0, 10.0],
                    "val_mae_curve": [22.0, 9.0],
                    "best_iteration": 2,
                    "best_val_mae": 9.0,
                    "mae_gap": -1.0,
                },
                {
                    "overall": {"mae": 11.0},
                    "train_mae_curve": [21.0, 16.0],
                    "val_mae_curve": [23.0, 11.0],
                    "best_iteration": 2,
                    "best_val_mae": 11.0,
                    "mae_gap": -5.0,
                },
            ],
        }

        with patch(
            "experiments.track_experiment.run_loss_diagnostic",
            return_value=fake_result,
        ):
            result = run_tracked_diagnostic(
                config={"data": {}},
                n_folds=2,
                cache_dir="cache",
                wandb_project="freight-rate-prediction",
                wandb_run_name="final-candidate-diagnostic",
                wandb_tags=["diagnostic"],
            )

        mock_init.assert_called_once()

        aggregate_call = mock_log.call_args_list[0][0][0]
        assert aggregate_call["mae_mean"] == fake_result["mean"]
        assert aggregate_call["n_rows"] == 1000
        assert aggregate_call["n_features"] == 14
        assert "runtime_seconds" in aggregate_call

        assert _mock_table.call_args.kwargs["columns"] == [
            "fold_index",
            "best_iteration",
            "best_val_mae",
            "mae_gap",
        ]
        _mock_table.return_value.add_data.assert_any_call(0, 2, 9.0, -1.0)
        _mock_table.return_value.add_data.assert_any_call(1, 2, 11.0, -5.0)

        mock_finish.assert_called_once()
        assert result["mean"] == fake_result["mean"]
        assert "runtime_seconds" in result


def test_run_tracked_experiment_skips_init_and_finish_when_run_already_active() -> None:
    with (
        patch.object(wandb, "init") as mock_init,
        patch.object(wandb, "log") as mock_log,
        patch.object(wandb, "finish") as mock_finish,
        patch.object(wandb, "Table") as _mock_table,
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
