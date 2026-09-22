from pathlib import Path
from unittest.mock import patch

import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

BASELINE_FEATURE_COLS = [
    "distance",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
]

CANDIDATE_FEATURE_COLS = [
    "distance",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "weight",
    "market_index",
    "quote_signal",
    "month_sin",
    "month_cos",
    "day_of_week",
    "equipment_Dry Van",
    "equipment_Flatbed",
    "equipment_Reefer",
]

FULL_FEATURE_FAMILIES = [
    "calendar",
    "equipment",
    "market_index_ema",
    "quote_signal_zscore",
]


def _read(name: str) -> dict:
    with open(CONFIGS_DIR / name) as f:
        return yaml.safe_load(f)


def _compose(overrides: list[str] | None = None):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS_DIR)):
        return compose(config_name="config", overrides=overrides or [])


def test_composed_config_matches_manually_assembled_config() -> None:
    composed = OmegaConf.to_container(_compose(), resolve=True)
    composed_config = {
        k: v for k, v in composed.items() if k not in ("n_folds", "cache_dir", "tracker")
    }

    candidate_overrides = _read("experiment/candidate.yaml")
    workflow_final_fit = _read("workflow/final_fit.yaml")

    manual_config = {
        "data": _read("data/default.yaml"),
        "cleaning": _read("cleaning/default.yaml"),
        "features": _read("features/default.yaml"),
        "model": _read("model/catboost.yaml"),
        "training": _read("training/default.yaml"),
    }
    manual_config["features"]["selected"] = candidate_overrides["features"]["selected"]
    manual_config["training"]["feature_cols"] = candidate_overrides["training"][
        "feature_cols"
    ]
    manual_config["workflow"] = workflow_final_fit

    assert composed_config == manual_config
    assert composed["n_folds"] == 5
    assert composed["cache_dir"] == "data/interim/cache"
    assert composed["tracker"] == "none"


def test_default_composition_selects_candidate_preset() -> None:
    composed = OmegaConf.to_container(_compose(), resolve=True)

    assert composed["features"]["selected"] == ["calendar", "equipment"]
    assert composed["training"]["feature_cols"] == CANDIDATE_FEATURE_COLS
    assert composed["model"]["family"] == "catboost"
    assert composed["model"]["params"]["random_state"] == 42
    assert composed["workflow"]["name"] == "final_fit"
    assert composed["tracker"] == "none"


def test_full_feature_preset_selects_declared_families_and_columns() -> None:
    composed = OmegaConf.to_container(
        _compose(overrides=["experiment=features_combined_v1"]), resolve=True
    )

    assert composed["features"]["selected"] == FULL_FEATURE_FAMILIES
    assert set(FULL_FEATURE_FAMILIES).issubset(set(composed["features"]["families"]))
    assert composed["training"]["feature_cols"][:5] == BASELINE_FEATURE_COLS
    for col in ["equipment_Dry Van", "equipment_Flatbed", "equipment_Reefer"]:
        assert col in composed["training"]["feature_cols"]
    assert "lane" not in composed["training"]["feature_cols"]


def test_hydra_entry_calls_run_experiment_with_composed_baseline_config() -> None:
    fake_result = {
        "metric": "mae",
        "mean": 12.0,
        "std": 1.5,
        "worst_fold": 14.0,
        "fold_metrics": [],
    }

    cfg = _compose(overrides=["model=lightgbm", "experiment=baseline", "workflow=none"])

    with patch(
        "experiments.hydra_entry.run_experiment", return_value=fake_result
    ) as mock_run_experiment:
        from experiments.hydra_entry import main

        result = main.__wrapped__(cfg)

    assert result == fake_result
    mock_run_experiment.assert_called_once()
    call_args, _call_kwargs = mock_run_experiment.call_args
    config_arg, n_folds_arg, cache_dir_arg = call_args

    assert n_folds_arg == 5
    assert cache_dir_arg == "data/interim/cache"
    assert config_arg["model"]["params"]["max_depth"] == -1
    assert config_arg["features"]["selected"] == []
    assert config_arg["training"]["feature_cols"] == BASELINE_FEATURE_COLS
    assert set(config_arg.keys()) == {"data", "cleaning", "features", "model", "training"}


def test_hydra_entry_calls_run_experiment_with_composed_full_feature_config() -> None:
    fake_result = {
        "metric": "mae",
        "mean": 9.0,
        "std": 1.0,
        "worst_fold": 10.5,
        "fold_metrics": [],
    }

    cfg = _compose(overrides=["experiment=features_combined_v1", "workflow=none"])

    with patch(
        "experiments.hydra_entry.run_experiment", return_value=fake_result
    ) as mock_run_experiment:
        from experiments.hydra_entry import main

        result = main.__wrapped__(cfg)

    assert result == fake_result
    config_arg = mock_run_experiment.call_args.args[0]
    assert config_arg["features"]["selected"] == FULL_FEATURE_FAMILIES
    assert "equipment_Dry Van" in config_arg["training"]["feature_cols"]


def _fake_diagnostic_result() -> dict:
    return {
        "mean": 119.5,
        "std": 15.2,
        "worst_fold": 141.4,
        "n_rows": 43147,
        "n_features": 14,
        "runtime_seconds": 7.9,
        "fold_diagnostics": [
            {"best_iteration": 200, "mae_gap": 40.0},
            {"best_iteration": 300, "mae_gap": 20.0},
        ],
    }


def _fake_final_fit_result() -> dict:
    return {
        "cleaner": object(),
        "feature_pipeline": object(),
        "trainer": object(),
        "raw_columns": [],
        "categorical_options": {},
        "train_curve": [],
        "n_rows": 43147,
        "n_features": 14,
        "in_sample_metrics": {"mae": 60.97},
        "runtime_seconds": 2.9,
    }


def test_hydra_entry_default_final_fit_does_not_touch_wandb() -> None:
    cfg = _compose()

    with (
        patch(
            "experiments.run_experiment.fit_final_model",
            return_value=_fake_final_fit_result(),
        ) as mock_fit,
        patch("experiments.track_experiment.run_tracked_final_fit") as mock_tracked,
        patch("serving.inference.PredictionBundle") as mock_bundle,
    ):
        from experiments.hydra_entry import main

        result = main.__wrapped__(cfg)

    mock_tracked.assert_not_called()
    mock_fit.assert_called_once()
    mock_bundle.return_value.save.assert_called_once_with("artifacts/model.joblib")
    assert result["n_rows"] == 43147
    assert "runtime_seconds" in result


def test_hydra_entry_tracker_wandb_opts_into_tracked_final_fit() -> None:
    cfg = _compose(overrides=["tracker=wandb"])

    with (
        patch(
            "experiments.track_experiment.run_tracked_final_fit",
            return_value=_fake_final_fit_result(),
        ) as mock_tracked,
        patch("experiments.run_experiment.fit_final_model") as mock_fit,
        patch("serving.inference.PredictionBundle"),
    ):
        from experiments.hydra_entry import main

        result = main.__wrapped__(cfg)

    mock_fit.assert_not_called()
    mock_tracked.assert_called_once()
    call_kwargs = mock_tracked.call_args.kwargs
    assert call_kwargs["wandb_run_name"] == "final-fit-catboost"
    assert call_kwargs["wandb_tags"] == ["final-fit", "catboost"]
    assert result["n_rows"] == 43147


def test_hydra_entry_tracker_wandb_opts_into_tracked_diagnostic() -> None:
    cfg = _compose(overrides=["workflow=diagnostic", "tracker=wandb"])

    with patch(
        "experiments.track_experiment.run_tracked_diagnostic",
        return_value=_fake_diagnostic_result(),
    ) as mock_diag:
        from experiments.hydra_entry import main

        result = main.__wrapped__(cfg)

    mock_diag.assert_called_once()
    assert result["mean"] == 119.5


def test_hydra_entry_diagnostic_workflow_maps_overrides_per_model_family() -> None:
    """workflow=diagnostic composed with a non-catboost model must map
    iterations/eval_metric onto that family's actual param names (e.g.
    LightGBM's n_estimators/metric), not silently pass catboost's.
    """
    cfg = _compose(overrides=["model=lightgbm", "experiment=baseline", "workflow=diagnostic"])

    with patch(
        "experiments.hydra_entry.run_loss_diagnostic",
        return_value=_fake_diagnostic_result(),
    ) as mock_diag:
        from experiments.hydra_entry import main

        main.__wrapped__(cfg)

    config_arg = mock_diag.call_args.args[0]
    assert config_arg["model"]["params"]["n_estimators"] == 1000
    assert config_arg["model"]["params"]["metric"] == "mae"
    assert config_arg["model"]["params"]["verbosity"] == -1
    assert "iterations" not in config_arg["model"]["params"]
    assert "eval_metric" not in config_arg["model"]["params"]


def test_hydra_entry_workflow_none_never_touches_holdout_or_wandb() -> None:
    """Sanity check that the plain run_experiment path (workflow=none,
    tracker defaulting to none) is the only thing exercised: neither
    tracked function is imported or called.
    """
    cfg = _compose(overrides=["workflow=none"])

    with (
        patch("experiments.hydra_entry.run_experiment", return_value={}) as mock_run,
        patch("experiments.track_experiment.run_tracked_diagnostic") as mock_diag,
        patch("experiments.track_experiment.run_tracked_experiment") as mock_tracked_exp,
    ):
        from experiments.hydra_entry import main

        main.__wrapped__(cfg)

    mock_run.assert_called_once()
    mock_diag.assert_not_called()
    mock_tracked_exp.assert_not_called()
