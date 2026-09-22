from unittest.mock import patch

import wandb


def test_launch_sweep_registers_sweep_and_returns_id() -> None:
    with patch.object(wandb, "sweep", return_value="sweep-xyz789") as mock_sweep:
        from experiments.launch_sweep import launch_sweep

        sweep_id = launch_sweep()

        mock_sweep.assert_called_once()
        call_args, call_kwargs = mock_sweep.call_args

        sweep_config = call_args[0] if call_args else call_kwargs["sweep"]
        assert sweep_config["method"] == "bayes"
        assert sweep_config["metric"]["name"] == "mae_mean"
        assert sweep_config["metric"]["goal"] == "minimize"
        assert "n_estimators" in sweep_config["parameters"]

        assert call_kwargs["project"] == "freight-rate-prediction"
        assert call_kwargs["prior_runs"] is None
        assert sweep_id == "sweep-xyz789"


def test_launch_sweep_forwards_prior_runs_to_warm_start_the_search() -> None:
    with patch.object(wandb, "sweep", return_value="sweep-warm123") as mock_sweep:
        from experiments.launch_sweep import launch_sweep

        sweep_id = launch_sweep(prior_runs=["run1", "run2"])

        call_kwargs = mock_sweep.call_args.kwargs
        assert call_kwargs["prior_runs"] == ["run1", "run2"]
        assert sweep_id == "sweep-warm123"


def test_launch_sweep_resumes_existing_sweep_id_without_creating_a_new_one() -> None:
    with patch.object(wandb, "sweep") as mock_sweep:
        from experiments.launch_sweep import launch_sweep

        sweep_id = launch_sweep(sweep_id="existing-sweep-id")

        mock_sweep.assert_not_called()
        assert sweep_id == "existing-sweep-id"
