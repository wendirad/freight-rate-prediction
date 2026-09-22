"""Orchestration-level regression coverage for run_experiment.

Guards against the original cross-validation leakage bug: cleaning and
feature engineering must be fit fresh, per fold, on that fold's training
rows only, never on the union of training and validation rows, and never
on October (the internal holdout).
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import experiments.run_experiment as run_experiment_mod
from data.cleaning import WeightCleaner
from data.folds import make_expanding_folds
from experiments.run_experiment import (
    evaluate_holdout,
    fit_final_model,
    run_experiment,
    run_loss_diagnostic,
)

OCTOBER = pd.Timestamp("2025-10-01")
N_FOLDS = 3


def test_feature_config_forwards_categorical_cols_from_training_config() -> None:
    training_cfg = {
        "feature_cols": ["distance", "lane"],
        "target_col": "posted_rate",
        "categorical_cols": ["lane"],
    }

    config = run_experiment_mod._feature_config(training_cfg)

    assert config.categorical_cols == ["lane"]


def test_feature_config_categorical_cols_defaults_to_none() -> None:
    training_cfg = {"feature_cols": ["distance"], "target_col": "posted_rate"}

    config = run_experiment_mod._feature_config(training_cfg)

    assert config.categorical_cols is None


def test_build_model_supports_linear_regression() -> None:
    from sklearn.linear_model import LinearRegression

    model = run_experiment_mod._build_model(
        {"family": "linear_regression", "params": {"fit_intercept": False}}
    )

    assert isinstance(model, LinearRegression)
    assert model.fit_intercept is False


def test_diagnostic_model_overrides_maps_catboost_param_names() -> None:
    workflow_cfg = {"iterations": 1000, "early_stopping_rounds": 50, "eval_metric": "MAE"}

    overrides = run_experiment_mod.diagnostic_model_overrides("catboost", workflow_cfg)

    assert overrides == {"iterations": 1000, "eval_metric": "MAE", "verbose": False}


def test_diagnostic_model_overrides_maps_lightgbm_param_names() -> None:
    workflow_cfg = {"iterations": 1000, "early_stopping_rounds": 50, "eval_metric": "MAE"}

    overrides = run_experiment_mod.diagnostic_model_overrides("lightgbm", workflow_cfg)

    assert overrides == {"n_estimators": 1000, "metric": "mae", "verbosity": -1}


def test_diagnostic_model_overrides_maps_xgboost_param_names() -> None:
    workflow_cfg = {"iterations": 1000, "early_stopping_rounds": 50, "eval_metric": "MAE"}

    overrides = run_experiment_mod.diagnostic_model_overrides("xgboost", workflow_cfg)

    assert overrides == {"n_estimators": 1000, "eval_metric": "mae", "verbosity": 0}


def _make_raw(n_dev: int = 40, n_holdout: int = 10) -> pd.DataFrame:
    dev_dates = pd.date_range("2025-01-01", periods=n_dev, freq="D")
    holdout_dates = pd.date_range("2025-10-01", periods=n_holdout, freq="D")
    dates = list(dev_dates) + list(holdout_dates)
    n = len(dates)

    return pd.DataFrame(
        {
            "load_id": range(n),
            "pickup": ["Chicago, IL"] * n,
            "delivery": ["Dallas, TX"] * n,
            "pickup_lat": np.linspace(41.8, 42.0, n),
            "pickup_lon": np.linspace(-87.6, -87.5, n),
            "delivery_lat": np.linspace(32.7, 32.9, n),
            "delivery_lon": np.linspace(-96.8, -96.6, n),
            "distance": np.linspace(900, 950, n),
            "equipment": ["Dry Van"] * n,
            "weight": np.linspace(10000, 40000, n),
            "date": dates,
            "market_index": np.linspace(1.0, 2.0, n),
            "quote_signal": np.linspace(0.1, 0.9, n),
            "posted_rate": np.linspace(1000.0, 2000.0, n),
        }
    )


def _config() -> dict:
    return {
        "data": {
            "date_col": "date",
            "split": {"test_start": "2025-10-01"},
            "raw_path": "unused",
            "required_columns": [],
        },
        "cleaning": {
            "type_caster": {},
            "category_normalizer": {},
            "weight_cleaner": {},
            "market_index_imputer": {},
            "duplicate_remover": {},
            "lane_distance_outlier_remover": {},
            "target_outlier_remover": {},
        },
        "features": {"selected": []},
        "model": {"family": "fake", "params": {}},
        "training": {
            "feature_cols": [
                "distance",
                "pickup_lat",
                "pickup_lon",
                "delivery_lat",
                "delivery_lon",
            ],
            "target_col": "posted_rate",
            "weight_col": None,
            "metric": "mae",
        },
    }


class _FakeModel:
    """Stand-in for LGBMRegressor: no external dependency, and its fit
    state (mean_) makes it trivial to assert what data it actually saw.
    """

    def fit(self, X, y, sample_weight=None, **kwargs):
        self.mean_ = float(np.asarray(y).mean())
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def _spy_on_pipeline_factory(monkeypatch, attr_name, fit_calls, transform_calls, kind):
    real_factory = getattr(run_experiment_mod, attr_name)
    created = []

    def spy_factory(*args, **kwargs):
        pipeline = real_factory(*args, **kwargs)
        created.append(pipeline)

        orig_fit_transform = pipeline.fit_transform
        orig_transform = pipeline.transform

        def fit_transform(df):
            fit_calls.append((kind, id(pipeline), len(df)))
            return orig_fit_transform(df)

        def transform(df, is_training=False):
            transform_calls.append((kind, id(pipeline), len(df), is_training))
            return orig_transform(df, is_training=is_training)

        pipeline.fit_transform = fit_transform
        pipeline.transform = transform
        return pipeline

    monkeypatch.setattr(run_experiment_mod, attr_name, spy_factory)
    return created


def test_run_experiment_fits_fold_local_pipelines_and_never_touches_october(
    monkeypatch,
) -> None:
    raw = _make_raw()
    monkeypatch.setattr(
        run_experiment_mod, "_load_raw", lambda config, cache_dir: raw
    )
    monkeypatch.setattr(run_experiment_mod, "_build_model", lambda cfg: _FakeModel())

    fit_calls: list[tuple[str, int, int]] = []
    transform_calls: list[tuple[str, int, int, bool]] = []

    created_cleaners = _spy_on_pipeline_factory(
        monkeypatch, "_build_cleaning_pipeline", fit_calls, transform_calls, "clean"
    )
    created_features = _spy_on_pipeline_factory(
        monkeypatch,
        "build_default_feature_pipeline",
        fit_calls,
        transform_calls,
        "feature",
    )

    with patch.object(run_experiment_mod, "evaluate_holdout") as mock_holdout:
        result = run_experiment(_config(), n_folds=N_FOLDS, cache_dir="unused")
        mock_holdout.assert_not_called()

    assert result["fold_metrics"]
    assert len(result["fold_metrics"]) == N_FOLDS

    # A fresh pipeline instance was constructed for every fold, and no
    # instance was reused between folds.
    assert len(created_cleaners) == N_FOLDS
    assert len(created_features) == N_FOLDS
    assert len({id(p) for p in created_cleaners}) == N_FOLDS
    assert len({id(p) for p in created_features}) == N_FOLDS

    dev = raw[raw["date"] < OCTOBER].reset_index(drop=True)
    expected_folds = make_expanding_folds(dev, N_FOLDS)

    clean_fits = [c for c in fit_calls if c[0] == "clean"]
    clean_transforms = [c for c in transform_calls if c[0] == "clean"]
    feature_fits = [c for c in fit_calls if c[0] == "feature"]
    feature_transforms = [c for c in transform_calls if c[0] == "feature"]

    assert len(clean_fits) == N_FOLDS
    assert len(clean_transforms) == N_FOLDS
    assert len(feature_fits) == N_FOLDS
    assert len(feature_transforms) == N_FOLDS

    for i, (train_idx, val_idx) in enumerate(expected_folds):
        # Each cleaner is fit on exactly that fold's raw training rows,
        # never on the union of training and validation.
        assert clean_fits[i][2] == len(train_idx)
        # The feature pipeline is fit on the fold's *cleaned* training
        # rows (training-only cleaning steps may drop rows), which can
        # only be at most as many rows as went into cleaning, and never
        # the training-plus-validation union.
        assert feature_fits[i][2] <= len(train_idx)
        assert feature_fits[i][2] < len(train_idx) + len(val_idx)
        # Validation rows are transformed with is_training=False, using
        # the pipeline already fitted on that fold's training rows.
        assert clean_transforms[i][2] == len(val_idx)
        assert clean_transforms[i][3] is False
        assert feature_transforms[i][2] == len(val_idx)
        assert feature_transforms[i][3] is False
        # The same pipeline instance that was fit is the one transformed.
        assert clean_fits[i][1] == clean_transforms[i][1]
        assert feature_fits[i][1] == feature_transforms[i][1]

    # October never enters fold fitting or evaluation: every fold's
    # training and validation rows come from indices within the dev pool,
    # which excludes October by construction (dev = raw[raw["date"] < OCTOBER]).
    last_train_idx, last_val_idx = expected_folds[-1]
    assert max(last_train_idx.max(), last_val_idx.max()) < len(dev)


def test_run_experiment_does_not_invoke_holdout_evaluation(monkeypatch) -> None:
    raw = _make_raw()
    monkeypatch.setattr(
        run_experiment_mod, "_load_raw", lambda config, cache_dir: raw
    )
    monkeypatch.setattr(run_experiment_mod, "_build_model", lambda cfg: _FakeModel())

    with patch.object(run_experiment_mod, "evaluate_holdout") as mock_holdout:
        run_experiment(_config(), n_folds=N_FOLDS, cache_dir="unused")

    mock_holdout.assert_not_called()


def test_evaluate_holdout_fits_once_on_dev_pool_and_scores_october_once(
    monkeypatch,
) -> None:
    raw = _make_raw()
    monkeypatch.setattr(
        run_experiment_mod, "_load_raw", lambda config, cache_dir: raw
    )
    monkeypatch.setattr(run_experiment_mod, "_build_model", lambda cfg: _FakeModel())

    fit_calls: list[tuple[str, int, int]] = []
    transform_calls: list[tuple[str, int, int, bool]] = []

    _spy_on_pipeline_factory(
        monkeypatch, "_build_cleaning_pipeline", fit_calls, transform_calls, "clean"
    )
    _spy_on_pipeline_factory(
        monkeypatch,
        "build_default_feature_pipeline",
        fit_calls,
        transform_calls,
        "feature",
    )

    result = evaluate_holdout(_config(), cache_dir="unused")

    dev = raw[raw["date"] < OCTOBER].reset_index(drop=True)
    holdout = raw[raw["date"] >= OCTOBER].reset_index(drop=True)

    clean_fits = [c for c in fit_calls if c[0] == "clean"]
    clean_transforms = [c for c in transform_calls if c[0] == "clean"]
    feature_fits = [c for c in fit_calls if c[0] == "feature"]
    feature_transforms = [c for c in transform_calls if c[0] == "feature"]

    assert len(clean_fits) == 1
    assert clean_fits[0][2] == len(dev)
    assert len(clean_transforms) == 1
    assert clean_transforms[0][2] == len(holdout)
    assert clean_transforms[0][3] is False

    assert len(feature_fits) == 1
    assert feature_fits[0][2] <= len(dev)
    assert feature_fits[0][2] < len(dev) + len(holdout)
    assert len(feature_transforms) == 1
    assert feature_transforms[0][2] == len(holdout)
    assert feature_transforms[0][3] is False

    assert "mae" in result


def test_fit_final_model_includes_october_after_holdout_evaluation(monkeypatch) -> None:
    raw = _make_raw()
    monkeypatch.setattr(run_experiment_mod, "_load_raw", lambda config, cache_dir: raw)
    monkeypatch.setattr(run_experiment_mod, "_build_model", lambda cfg: _FakeModel())

    fit_calls: list[tuple[str, int, int]] = []
    transform_calls: list[tuple[str, int, int, bool]] = []
    _spy_on_pipeline_factory(
        monkeypatch, "_build_cleaning_pipeline", fit_calls, transform_calls, "clean"
    )
    _spy_on_pipeline_factory(
        monkeypatch,
        "build_default_feature_pipeline",
        fit_calls,
        transform_calls,
        "feature",
    )

    result = fit_final_model(_config(), cache_dir="unused")

    clean_fits = [call for call in fit_calls if call[0] == "clean"]
    assert clean_fits[0][2] == len(raw)
    assert result["n_rows"] == len(raw)


class _FakeLGBMLikeModel:
    """Mimics enough of LGBMRegressor's fit signature/attributes (eval_set,
    eval_names, callbacks, evals_result_, best_iteration_) for
    run_loss_diagnostic to exercise without needing real lightgbm training.
    """

    def __init__(self, n_estimators: int = 5) -> None:
        self.n_estimators = n_estimators

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        eval_set=None,
        eval_names=None,
        callbacks=None,
    ):
        self.mean_ = float(np.asarray(y).mean())
        self.best_iteration_ = min(3, self.n_estimators)
        if eval_set is not None:
            names = eval_names or [f"set_{i}" for i in range(len(eval_set))]
            self.evals_result_ = {
                name: {"l1": [10.0 / (i + 1) for i in range(self.n_estimators)]}
                for name in names
            }
        else:
            self.evals_result_ = {}
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def test_run_loss_diagnostic_captures_curves_and_best_iteration(monkeypatch) -> None:
    raw = _make_raw()
    monkeypatch.setattr(run_experiment_mod, "_load_raw", lambda config, cache_dir: raw)
    monkeypatch.setattr(
        run_experiment_mod, "_build_model", lambda cfg: _FakeLGBMLikeModel()
    )

    config = _config()
    config["training"]["early_stopping_rounds"] = 2

    result = run_loss_diagnostic(config, n_folds=N_FOLDS, cache_dir="unused")

    assert len(result["fold_diagnostics"]) == N_FOLDS
    for diag in result["fold_diagnostics"]:
        assert diag["best_iteration"] == 3
        assert len(diag["train_mae_curve"]) == 5
        assert len(diag["val_mae_curve"]) == 5
        assert diag["best_val_mae"] == min(diag["val_mae_curve"])
    assert "mean" in result and "std" in result and "worst_fold" in result


def test_run_loss_diagnostic_computes_mae_gap_and_reports_rows_and_features(
    monkeypatch,
) -> None:
    raw = _make_raw()
    monkeypatch.setattr(run_experiment_mod, "_load_raw", lambda config, cache_dir: raw)
    monkeypatch.setattr(
        run_experiment_mod, "_build_model", lambda cfg: _FakeLGBMLikeModel()
    )

    config = _config()
    config["training"]["early_stopping_rounds"] = 2

    result = run_loss_diagnostic(config, n_folds=N_FOLDS, cache_dir="unused")

    dev, _holdout = run_experiment_mod._split_dev_holdout(
        config, run_experiment_mod._load_raw(config, "unused")
    )
    assert result["n_rows"] == len(dev)
    assert result["n_features"] == len(config["training"]["feature_cols"])

    for diag in result["fold_diagnostics"]:
        # _FakeLGBMLikeModel's best_val_mae is the curve's global min, while
        # mae_gap compares against the value at best_iteration specifically
        # (index 2, since best_iteration is hardcoded to 3) — deterministic
        # given the fixture's fixed "l1" curve formula, but not zero, since
        # the curve's minimum isn't at that index.
        expected_gap = min(diag["val_mae_curve"]) - diag["train_mae_curve"][2]
        assert diag["mae_gap"] == pytest.approx(expected_gap)


def test_run_experiment_preserves_existing_behavior_without_early_stopping(
    monkeypatch,
) -> None:
    """Step 5: run_experiment (used by sweeps) must not pass eval_df/early
    stopping to Trainer.train even when a config happens to carry
    early_stopping_rounds; only run_loss_diagnostic does that.
    """
    raw = _make_raw()
    monkeypatch.setattr(run_experiment_mod, "_load_raw", lambda config, cache_dir: raw)

    seen_eval_df = []
    real_train = run_experiment_mod.Trainer.train

    def spy_train(self, df, eval_df=None, early_stopping_rounds=None):
        seen_eval_df.append(eval_df)
        return real_train(self, df, eval_df=eval_df, early_stopping_rounds=early_stopping_rounds)

    monkeypatch.setattr(run_experiment_mod.Trainer, "train", spy_train)
    monkeypatch.setattr(run_experiment_mod, "_build_model", lambda cfg: _FakeModel())

    config = _config()
    config["training"]["early_stopping_rounds"] = 50

    run_experiment(config, n_folds=N_FOLDS, cache_dir="unused")

    assert all(eval_df is None for eval_df in seen_eval_df)


def test_weight_cleaner_state_differs_between_train_only_and_train_plus_val_fit() -> (
    None
):
    """Focused state test: fitting on training rows only must learn
    different state than fitting on the union of training and validation
    rows, demonstrating why a shared/global fit would leak.
    """
    train = pd.DataFrame({"equipment": ["Dry Van"] * 3, "weight": [100.0, 200.0, 300.0]})
    val = pd.DataFrame({"equipment": ["Dry Van"] * 2, "weight": [50000.0, 60000.0]})

    train_only = WeightCleaner().fit(train)
    train_plus_val = WeightCleaner().fit(pd.concat([train, val], ignore_index=True))

    assert train_only.global_median_ != train_plus_val.global_median_
    assert train_only.equipment_medians_ != train_plus_val.equipment_medians_


def test_run_experiment_caches_only_raw_data_and_reflects_source_changes(
    monkeypatch, tmp_path
) -> None:
    """The experiment path must cache only the loaded raw dataframe, keyed
    in part on a source-file fingerprint, never cleaned or featured output.
    """
    raw_csv = tmp_path / "raw.csv"
    raw = _make_raw()
    raw.to_csv(raw_csv, index=False)

    monkeypatch.setattr(run_experiment_mod, "_build_model", lambda cfg: _FakeModel())

    config = _config()
    config["data"]["raw_path"] = str(raw_csv)
    config["data"]["required_columns"] = list(raw.columns)

    cache_dir = tmp_path / "cache"

    load_calls = {"n": 0}
    real_load = run_experiment_mod.DataLoader.load

    def counting_load(self):
        load_calls["n"] += 1
        return real_load(self)

    monkeypatch.setattr(run_experiment_mod.DataLoader, "load", counting_load)

    run_experiment(config, n_folds=N_FOLDS, cache_dir=str(cache_dir))
    assert load_calls["n"] == 1
    cached_files_after_first_run = set(cache_dir.glob("*.parquet"))
    assert len(cached_files_after_first_run) == 1

    # Same source file: the raw dataframe is reused from cache, not reloaded.
    run_experiment(config, n_folds=N_FOLDS, cache_dir=str(cache_dir))
    assert load_calls["n"] == 1
    assert set(cache_dir.glob("*.parquet")) == cached_files_after_first_run

    # Touching the source file changes its fingerprint, invalidating the
    # cache entry and forcing a reload.
    raw.to_csv(raw_csv, index=False)
    run_experiment(config, n_folds=N_FOLDS, cache_dir=str(cache_dir))
    assert load_calls["n"] == 2
    assert len(set(cache_dir.glob("*.parquet"))) == 2
