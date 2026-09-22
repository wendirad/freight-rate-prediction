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
        assert sweep_id == "sweep-xyz789"
