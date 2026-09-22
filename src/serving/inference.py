"""Serializable, preprocessing-aware inference bundle."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from experiments.run_experiment import (
    DEFAULT_CACHE_DIR,
    _build_cleaning_pipeline,
    _build_model,
    _feature_config,
    _load_raw,
    _split_dev_holdout,
)
from features.engineering import build_default_feature_pipeline
from models.trainer import Trainer


@dataclass
class PredictionBundle:
    """The fitted objects and schema required to reproduce model inputs."""

    cleaner: Any
    feature_pipeline: Any
    trainer: Trainer
    input_columns: list[str]
    prediction_col: str = "predicted_posted_rate"
    categorical_options: dict[str, list[str]] = field(default_factory=dict)

    def predict(self, raw: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in self.input_columns if column not in raw.columns]
        if missing:
            raise ValueError(f"missing required columns: {missing}")

        source = raw.copy()
        cleaned = self.cleaner.transform(source, is_training=False)
        featured = self.feature_pipeline.transform(cleaned, is_training=False)

        result = source.copy()
        result[self.prediction_col] = self.trainer.predict(featured)
        return result

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output)

    @staticmethod
    def load(path: str | Path) -> PredictionBundle:
        return joblib.load(path)


def build_prediction_bundle(
    config: dict[str, Any],
    cache_dir: str = DEFAULT_CACHE_DIR,
    include_holdout: bool = False,
) -> PredictionBundle:
    """Fit one deployable bundle after model selection is complete.

    By default the configured holdout remains untouched. Set include_holdout
    only after the final holdout evaluation and model-selection decisions.
    """

    raw = _load_raw(config, cache_dir)
    dev, _holdout = _split_dev_holdout(config, raw)
    training_raw = raw if include_holdout else dev

    cleaner = _build_cleaning_pipeline(config["cleaning"])
    cleaned = cleaner.fit_transform(training_raw)

    feature_pipeline = build_default_feature_pipeline(config["features"]["selected"])
    featured = feature_pipeline.fit_transform(cleaned)

    trainer = Trainer(
        _build_model(config["model"]),
        _feature_config(config["training"]),
    ).train(featured)

    target_col = config["training"]["target_col"]
    input_columns = [column for column in raw.columns if column != target_col]
    categorical_options = {
        column: sorted(cleaned[column].dropna().astype(str).unique().tolist())
        for column in ("equipment",)
        if column in cleaned.columns
    }

    return PredictionBundle(
        cleaner=cleaner,
        feature_pipeline=feature_pipeline,
        trainer=trainer,
        input_columns=input_columns,
        categorical_options=categorical_options,
    )
