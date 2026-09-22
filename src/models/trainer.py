"""Model-agnostic training and evaluation for the freight rate model."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd


class Estimator(Protocol):
    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray | None = None
    ) -> Any: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


@dataclass
class FeatureConfig:
    feature_cols: list[str]
    target_col: str = "posted_rate"
    weight_col: str | None = None
    group_col: str | None = "equipment"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    y_pred_clipped = np.clip(y_pred, a_min=0, a_max=None)
    rmsle = np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred_clipped)) ** 2))

    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {"mae": mae, "rmse": rmse, "rmsle": rmsle, "mape": mape}


def compute_grouped_metrics(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    group_col: str,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for group_value, subset in df.groupby(group_col):
        metrics = compute_metrics(subset[y_true_col].values, subset[y_pred_col].values)
        metrics["n"] = len(subset)
        results[str(group_value)] = metrics
    return results


class Trainer:
    """Wraps any estimator following the fit/predict sklearn interface.
    Owns feature selection, sample weighting, training, and evaluation.
    Has no knowledge of experiment tracking, logging is wired in from
    outside by whoever calls train and validate.
    """

    def __init__(self, model: Estimator, config: FeatureConfig) -> None:
        self.model = model
        self.config = config
        self.is_fitted_ = False

    def _select_xy(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        X = df[self.config.feature_cols]
        y = df[self.config.target_col]
        return X, y

    def _select_weights(self, df: pd.DataFrame) -> np.ndarray | None:
        if self.config.weight_col is None:
            return None
        return df[self.config.weight_col].values

    def train(
        self,
        df: pd.DataFrame,
        eval_df: pd.DataFrame | None = None,
        early_stopping_rounds: int | None = None,
    ) -> Trainer:
        X, y = self._select_xy(df)
        sample_weight = self._select_weights(df)

        fit_kwargs: dict[str, Any] = {}
        fit_params = inspect.signature(self.model.fit).parameters
        if eval_df is not None and "eval_set" in fit_params:
            X_val, y_val = self._select_xy(eval_df)
            fit_kwargs["eval_set"] = [(X, y), (X_val, y_val)]
            if "eval_names" in fit_params:
                fit_kwargs["eval_names"] = ["training", "valid"]
            if early_stopping_rounds is not None and "callbacks" in fit_params:
                import lightgbm as lgb

                fit_kwargs["callbacks"] = [lgb.early_stopping(early_stopping_rounds)]

        self.model.fit(X, y, sample_weight=sample_weight, **fit_kwargs)
        self.is_fitted_ = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("model must be trained before predict")
        X = df[self.config.feature_cols]
        return self.model.predict(X)

    def validate(self, df: pd.DataFrame) -> dict[str, Any]:
        predictions = self.predict(df)
        y_true = df[self.config.target_col].values

        overall = compute_metrics(y_true, predictions)

        grouped: dict[str, dict[str, float]] = {}
        if self.config.group_col is not None and self.config.group_col in df.columns:
            eval_df = df.copy()
            eval_df["_prediction"] = predictions
            grouped = compute_grouped_metrics(
                eval_df, self.config.target_col, "_prediction", self.config.group_col
            )

        return {"overall": overall, "by_group": grouped}


def build_equipment_sample_weights(
    df: pd.DataFrame, equipment_col: str = "equipment"
) -> np.ndarray:
    counts = df[equipment_col].value_counts()
    weights = (1 / counts) * counts.min()
    return df[equipment_col].map(weights).values
