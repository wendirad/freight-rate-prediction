"""End-to-end experiment runner: load, clean, engineer features, run
temporal CV folds, train and evaluate each fold, aggregate metrics.

This is the one function a W&B sweep agent or a manual baseline run calls.
No W&B or Hydra wiring here; run_experiment takes and returns plain dicts
so it is callable and testable on its own.
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
    raise ValueError(f"unknown model family: {family}")


def _load_and_clean(
    config: dict[str, Any], cache_dir: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_cfg = config["data"]
    schema = SchemaConfig(
        required_columns=data_cfg["required_columns"], date_col=data_cfg["date_col"]
    )
    loader = DataLoader(data_cfg["raw_path"], schema=schema)
    df = loader.load()

    test_start = pd.Timestamp(data_cfg["split"]["test_start"])
    date_col = data_cfg["date_col"]
    train_pool = df[df[date_col] < test_start].reset_index(drop=True)
    test = df[df[date_col] >= test_start].reset_index(drop=True)

    cleaning_cfg = config["cleaning"]
    clean_key = config_hash(
        {"stage": "clean", "cleaning": cleaning_cfg, "raw_path": data_cfg["raw_path"]}
    )
    cleaned = cached_dataframe_call(
        lambda: _build_cleaning_pipeline(cleaning_cfg).fit_transform(train_pool),
        clean_key,
        cache_dir,
    )
    return cleaned, test


def _engineer_features(
    config: dict[str, Any], cleaned: pd.DataFrame, cache_dir: str
) -> pd.DataFrame:
    families = config["features"]["selected"]
    feature_key = config_hash(
        {
            "stage": "features",
            "families": sorted(families),
            "cleaning": config["cleaning"],
            "raw_path": config["data"]["raw_path"],
        }
    )
    return cached_dataframe_call(
        lambda: build_default_feature_pipeline(families).fit_transform(cleaned),
        feature_key,
        cache_dir,
    )


def run_experiment(
    config: dict[str, Any], n_folds: int = 5, cache_dir: str = DEFAULT_CACHE_DIR
) -> dict[str, Any]:
    """Runs the full load -> clean -> engineer -> CV -> train -> evaluate
    loop for one config, and returns a single aggregated result dict.

    config must have "data", "cleaning", "features" (with a "selected" list
    naming which configs/features.yaml families to use), "model", and
    "training" keys, matching the shape load_config returns.
    """
    cleaned, _test = _load_and_clean(config, cache_dir)
    featured = _engineer_features(config, cleaned, cache_dir)

    training_cfg = config["training"]
    feature_config = FeatureConfig(
        feature_cols=training_cfg["feature_cols"],
        target_col=training_cfg["target_col"],
        weight_col=training_cfg.get("weight_col"),
    )
    metric = training_cfg.get("metric", "mae")

    folds = make_expanding_folds(featured, n_folds)

    fold_results = []
    for train_idx, val_idx in folds:
        fold_train = featured.iloc[train_idx].reset_index(drop=True)
        fold_val = featured.iloc[val_idx].reset_index(drop=True)

        model = _build_model(config["model"])
        trainer = Trainer(model, feature_config).train(fold_train)
        report = Evaluator(trainer).evaluate(fold_val)
        fold_results.append(report.overall)

    scores = np.array([r[metric] for r in fold_results])

    return {
        "fold_metrics": fold_results,
        "metric": metric,
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "worst_fold": float(scores.max()),
    }
