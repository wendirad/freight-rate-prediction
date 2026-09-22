"""End-to-end experiment runner: load, split into a dev pool and an
October holdout, run temporal CV folds with fold-local cleaning and
feature fitting, train and evaluate each fold, aggregate metrics.

This is the one function a W&B sweep agent or a manual baseline run calls.
No W&B or Hydra wiring here; run_experiment takes and returns plain dicts
so it is callable and testable on its own.

October (config["data"]["split"]["test_start"] onward) is the internal
holdout and is never touched by run_experiment: only evaluate_holdout,
which must be invoked explicitly, ever fits or scores on it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from data.cleaning import (
    CategoryNormalizer,
    CleaningPipeline,
    DuplicateRemover,
    LaneDistanceOutlierRemover,
    MarketIndexImputer,
    TargetOutlierRemover,
    TypeCaster,
    WeightCleaner,
)
from data.folds import make_expanding_folds
from data.loading import DataLoader, SchemaConfig
from features.engineering import build_default_feature_pipeline
from models.evaluator import Evaluator
from models.trainer import FeatureConfig, Trainer
from utils.caching import cached_dataframe_call, config_hash

DEFAULT_CACHE_DIR = "data/interim/cache"


def load_config(
    configs_dir: str | Path = "configs", model_family: str = "lightgbm"
) -> dict[str, Any]:
    """Reads the yaml files under configs_dir into a single plain config dict."""
    configs_dir = Path(configs_dir)

    def _read(name: str) -> dict:
        with open(configs_dir / name) as f:
            return yaml.safe_load(f)

    return {
        "data": _read("data.yaml"),
        "cleaning": _read("cleaning.yaml"),
        "features": _read("features.yaml"),
        "model": _read(f"model/{model_family}.yaml"),
        "training": _read("training.yaml"),
    }


def _build_cleaning_pipeline(cleaning_cfg: dict[str, Any]) -> CleaningPipeline:
    return CleaningPipeline(
        [
            TypeCaster(**cleaning_cfg["type_caster"]),
            CategoryNormalizer(**cleaning_cfg["category_normalizer"]),
            WeightCleaner(**cleaning_cfg["weight_cleaner"]),
            MarketIndexImputer(**cleaning_cfg["market_index_imputer"]),
            DuplicateRemover(**cleaning_cfg["duplicate_remover"]),
            LaneDistanceOutlierRemover(**cleaning_cfg["lane_distance_outlier_remover"]),
            TargetOutlierRemover(**cleaning_cfg["target_outlier_remover"]),
        ]
    )


def _build_model(model_cfg: dict[str, Any]):
    family = model_cfg["family"]
    params = model_cfg.get("params", {})

    if family == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(**params)
    if family == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(**params)
    if family == "catboost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(**params)
    raise ValueError(f"unknown model family: {family}")


_ITERATIONS_PARAM_BY_FAMILY = {
    "catboost": "iterations",
    "lightgbm": "n_estimators",
    "xgboost": "n_estimators",
}
_EVAL_METRIC_PARAM_BY_FAMILY = {
    "catboost": "eval_metric",
    "lightgbm": "metric",
    "xgboost": "eval_metric",
}
_EVAL_METRIC_VALUE_BY_FAMILY = {
    "catboost": "MAE",
    "lightgbm": "mae",
    "xgboost": "mae",
}
# Silences each library's own per-iteration/per-bin training log spam
# ("Total Bins ...", "[LightGBM] [Info] ..."), so the diagnostic workflow's
# own curve summary (see hydra_entry._summarize_diagnostic) is the signal
# the user sees, not library internals.
_QUIET_PARAMS_BY_FAMILY = {
    "catboost": {"verbose": False},
    "lightgbm": {"verbosity": -1},
    "xgboost": {"verbosity": 0},
}


def diagnostic_model_overrides(family: str, workflow_cfg: dict[str, Any]) -> dict[str, Any]:
    """Maps the generic `iterations`/`eval_metric` knobs from
    configs/workflow/diagnostic.yaml onto the param names and value
    spellings the given model family's estimator actually expects, so the
    diagnostic workflow composes with any model= override. Also forces
    that family's quiet-training param, so only this project's own curve
    summary is printed, not the library's internal training log.
    """
    iterations_param = _ITERATIONS_PARAM_BY_FAMILY.get(family, "n_estimators")
    metric_param = _EVAL_METRIC_PARAM_BY_FAMILY.get(family, "eval_metric")
    metric_value = _EVAL_METRIC_VALUE_BY_FAMILY.get(family, workflow_cfg["eval_metric"])
    return {
        iterations_param: workflow_cfg["iterations"],
        metric_param: metric_value,
        **_QUIET_PARAMS_BY_FAMILY.get(family, {}),
    }


def _feature_config(training_cfg: dict[str, Any]) -> FeatureConfig:
    return FeatureConfig(
        feature_cols=training_cfg["feature_cols"],
        target_col=training_cfg["target_col"],
        weight_col=training_cfg.get("weight_col"),
        categorical_cols=training_cfg.get("categorical_cols"),
    )


def _source_fingerprint(path: Path) -> str | None:
    """A cheap signal sufficient to invalidate the raw-data cache when the
    underlying file changes: its size and modification time. Returns None
    if the file does not exist, which itself changes the cache key.
    """
    if not path.exists():
        return None
    stat = path.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def _load_raw(config: dict[str, Any], cache_dir: str) -> pd.DataFrame:
    """Loads and schema-validates the raw source dataframe (chronologically
    sorted, untouched by cleaning or feature engineering), cached only on
    the data-loading config, schema, source path, and a source-file
    fingerprint.
    """
    data_cfg = config["data"]
    schema = SchemaConfig(
        required_columns=data_cfg["required_columns"], date_col=data_cfg["date_col"]
    )
    raw_path = Path(data_cfg["raw_path"])

    raw_key = config_hash(
        {
            "stage": "raw",
            "data": data_cfg,
            "schema": {
                "required_columns": schema.required_columns,
                "date_col": schema.date_col,
            },
            "source_path": str(raw_path),
            "source_fingerprint": _source_fingerprint(raw_path),
        }
    )
    loader = DataLoader(raw_path, schema=schema)
    return cached_dataframe_call(loader.load, raw_key, cache_dir)


def _split_dev_holdout(
    config: dict[str, Any], df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the raw, chronologically sorted dataframe in memory into the
    pre-October development pool (used for cross-validation) and the
    October-onward internal holdout.
    """
    data_cfg = config["data"]
    date_col = data_cfg["date_col"]
    test_start = pd.Timestamp(data_cfg["split"]["test_start"])

    dev = df[df[date_col] < test_start].reset_index(drop=True)
    holdout = df[df[date_col] >= test_start].reset_index(drop=True)
    return dev, holdout


def _fit_and_score_fold(
    config: dict[str, Any],
    fold_train_raw: pd.DataFrame,
    fold_val_raw: pd.DataFrame,
    feature_config: FeatureConfig,
) -> dict[str, float]:
    """Fits a fresh cleaning pipeline and a fresh feature pipeline on this
    fold's raw training rows only, transforms the fold's validation rows
    with those fitted pipelines (is_training=False), trains a fresh model,
    and evaluates it. No state crosses fold boundaries.
    """
    cleaner = _build_cleaning_pipeline(config["cleaning"])
    cleaned_train = cleaner.fit_transform(fold_train_raw)
    cleaned_val = cleaner.transform(fold_val_raw, is_training=False)

    families = config["features"]["selected"]
    feature_pipeline = build_default_feature_pipeline(families)
    featured_train = feature_pipeline.fit_transform(cleaned_train)
    featured_val = feature_pipeline.transform(cleaned_val, is_training=False)

    model = _build_model(config["model"])
    trainer = Trainer(model, feature_config).train(featured_train)
    report = Evaluator(trainer).evaluate(featured_val)
    return report.overall


def run_experiment(
    config: dict[str, Any], n_folds: int = 5, cache_dir: str = DEFAULT_CACHE_DIR
) -> dict[str, Any]:
    """Runs the full load -> split -> per-fold (clean -> engineer -> train
    -> evaluate) cross-validation loop for one config, and returns a single
    aggregated result dict.

    config must have "data", "cleaning", "features" (with a "selected" list
    naming which configs/features.yaml families to use), "model", and
    "training" keys, matching the shape load_config returns.

    Fold assignment happens on the raw, chronologically sorted, pre-October
    development pool. Every fold fits its own cleaning and feature
    pipelines on that fold's training rows only; nothing is fit before
    entering the fold loop, and no fitted pipeline is reused between folds.
    October is excluded before folds are created and is never evaluated
    here; use evaluate_holdout for that, explicitly.
    """
    raw = _load_raw(config, cache_dir)
    dev, _holdout = _split_dev_holdout(config, raw)

    training_cfg = config["training"]
    feature_config = _feature_config(training_cfg)
    metric = training_cfg.get("metric", "mae")

    folds = make_expanding_folds(dev, n_folds)

    fold_results = []
    for train_idx, val_idx in folds:
        fold_train_raw = dev.iloc[train_idx].reset_index(drop=True)
        fold_val_raw = dev.iloc[val_idx].reset_index(drop=True)
        fold_results.append(
            _fit_and_score_fold(config, fold_train_raw, fold_val_raw, feature_config)
        )

    scores = np.array([r[metric] for r in fold_results])

    return {
        "fold_metrics": fold_results,
        "metric": metric,
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "worst_fold": float(scores.max()),
    }


def _fit_and_diagnose_fold(
    config: dict[str, Any],
    fold_train_raw: pd.DataFrame,
    fold_val_raw: pd.DataFrame,
    feature_config: FeatureConfig,
    early_stopping_rounds: int | None,
) -> dict[str, Any]:
    """Like _fit_and_score_fold, but passes validation data to Trainer.train
    (with early stopping, if configured) and captures the per-iteration
    train/validation loss curve alongside the usual evaluation report.
    """
    cleaner = _build_cleaning_pipeline(config["cleaning"])
    cleaned_train = cleaner.fit_transform(fold_train_raw)
    cleaned_val = cleaner.transform(fold_val_raw, is_training=False)

    families = config["features"]["selected"]
    feature_pipeline = build_default_feature_pipeline(families)
    featured_train = feature_pipeline.fit_transform(cleaned_train)
    featured_val = feature_pipeline.transform(cleaned_val, is_training=False)

    model = _build_model(config["model"])
    trainer = Trainer(model, feature_config).train(
        featured_train,
        eval_df=featured_val,
        early_stopping_rounds=early_stopping_rounds,
    )
    report = Evaluator(trainer).evaluate(featured_val)

    evals_result = getattr(model, "evals_result_", None) or {}
    train_curve: list[float] = []
    val_curve: list[float] = []
    if evals_result:
        train_key = "training" if "training" in evals_result else next(iter(evals_result))
        val_key = next(k for k in evals_result if k != train_key)
        metric_key = next(iter(evals_result[train_key]))
        train_curve = list(evals_result[train_key][metric_key])
        val_curve = list(evals_result[val_key][metric_key])

    best_iteration = getattr(model, "best_iteration_", None)
    best_val_mae = min(val_curve) if val_curve else report.overall.get("mae")

    mae_gap = None
    if train_curve and best_iteration:
        curve_idx = min(best_iteration, len(train_curve)) - 1
        mae_gap = best_val_mae - train_curve[curve_idx]

    return {
        "overall": report.overall,
        "train_mae_curve": train_curve,
        "val_mae_curve": val_curve,
        "best_iteration": best_iteration,
        "best_val_mae": best_val_mae,
        "mae_gap": mae_gap,
    }


def run_loss_diagnostic(
    config: dict[str, Any], n_folds: int = 5, cache_dir: str = DEFAULT_CACHE_DIR
) -> dict[str, Any]:
    """Like run_experiment, but for each fold also feeds validation data to
    Trainer.train (with training_cfg["early_stopping_rounds"], if set) and
    captures that fold's train/validation MAE curve and best iteration, for
    diagnosing overfitting and choosing n_estimators / early-stopping
    patience. Never called from run_experiment; must be invoked explicitly.
    """
    raw = _load_raw(config, cache_dir)
    dev, _holdout = _split_dev_holdout(config, raw)

    training_cfg = config["training"]
    feature_config = _feature_config(training_cfg)
    metric = training_cfg.get("metric", "mae")
    early_stopping_rounds = training_cfg.get("early_stopping_rounds")

    folds = make_expanding_folds(dev, n_folds)

    fold_diagnostics = []
    for train_idx, val_idx in folds:
        fold_train_raw = dev.iloc[train_idx].reset_index(drop=True)
        fold_val_raw = dev.iloc[val_idx].reset_index(drop=True)
        fold_diagnostics.append(
            _fit_and_diagnose_fold(
                config, fold_train_raw, fold_val_raw, feature_config, early_stopping_rounds
            )
        )

    scores = np.array([d["overall"][metric] for d in fold_diagnostics])

    return {
        "fold_metrics": [d["overall"] for d in fold_diagnostics],
        "fold_diagnostics": fold_diagnostics,
        "metric": metric,
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "worst_fold": float(scores.max()),
        "n_rows": len(dev),
        "n_features": len(feature_config.feature_cols),
    }


def evaluate_holdout(
    config: dict[str, Any], cache_dir: str = DEFAULT_CACHE_DIR
) -> dict[str, Any]:
    """Explicit, separately invoked October holdout evaluation.

    Fits a fresh cleaning pipeline and a fresh feature pipeline on the
    entire pre-October development pool, transforms October with
    is_training=False (never refitting), fits one fresh model on the
    transformed development pool, and evaluates it once on transformed
    October.

    Not called by run_experiment, which is used by sweeps: automatically
    returning holdout metrics from that path would expose the holdout
    during tuning. Call this only when final model and feature selection
    are complete.
    """
    raw = _load_raw(config, cache_dir)
    dev, holdout = _split_dev_holdout(config, raw)

    cleaner = _build_cleaning_pipeline(config["cleaning"])
    cleaned_dev = cleaner.fit_transform(dev)
    cleaned_holdout = cleaner.transform(holdout, is_training=False)

    families = config["features"]["selected"]
    feature_pipeline = build_default_feature_pipeline(families)
    featured_dev = feature_pipeline.fit_transform(cleaned_dev)
    featured_holdout = feature_pipeline.transform(cleaned_holdout, is_training=False)

    training_cfg = config["training"]
    feature_config = _feature_config(training_cfg)
    metric = training_cfg.get("metric", "mae")

    model = _build_model(config["model"])
    trainer = Trainer(model, feature_config).train(featured_dev)
    report = Evaluator(trainer).evaluate(featured_holdout)

    return {"metric": metric, **report.overall}


def fit_final_model(
    config: dict[str, Any], cache_dir: str = DEFAULT_CACHE_DIR
) -> dict[str, Any]:
    """Fits one model on the entire pre-October dev pool: no per-fold CV,
    no October access, no early stopping. config["model"]["params"] is
    expected to already carry a fixed iteration count chosen from a prior
    diagnostic run (run_loss_diagnostic / the diagnostic workflow), not a
    ceiling to search over.

    Captures a self-referential training curve for logging purposes only:
    Trainer is given the same fitted data as both train and eval, since
    once every dev row goes into training there is no held-out split left
    to evaluate against. This is not a generalization metric.
    """
    raw = _load_raw(config, cache_dir)
    dev, _holdout = _split_dev_holdout(config, raw)

    cleaner = _build_cleaning_pipeline(config["cleaning"])
    cleaned = cleaner.fit_transform(dev)

    families = config["features"]["selected"]
    feature_pipeline = build_default_feature_pipeline(families)
    featured = feature_pipeline.fit_transform(cleaned)

    feature_config = _feature_config(config["training"])
    model = _build_model(config["model"])
    trainer = Trainer(model, feature_config).train(featured, eval_df=featured)
    report = Evaluator(trainer).evaluate(featured)

    evals_result = getattr(model, "evals_result_", None) or {}
    train_curve: list[float] = []
    if evals_result:
        first_key = next(iter(evals_result))
        metric_key = next(iter(evals_result[first_key]))
        train_curve = list(evals_result[first_key][metric_key])

    target_col = config["training"]["target_col"]
    categorical_options = {
        column: sorted(cleaned[column].dropna().astype(str).unique().tolist())
        for column in ("equipment",)
        if column in cleaned.columns
    }

    return {
        "cleaner": cleaner,
        "feature_pipeline": feature_pipeline,
        "trainer": trainer,
        "train_curve": train_curve,
        "n_rows": len(dev),
        "n_features": len(feature_config.feature_cols),
        "in_sample_metrics": report.overall,
        "raw_columns": [c for c in raw.columns if c != target_col],
        "categorical_options": categorical_options,
    }
