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
    categorical_cols: list[str] | None = None
    target_transform: str = "direct"
    distance_col: str = "distance"
    date_col: str = "date"


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
        self.date_origin_: pd.Timestamp | None = None
        self.date_trend_intercept_: float | None = None
        self.date_trend_slope_: float | None = None

    def _positive_distance(self, df: pd.DataFrame) -> pd.Series:
        distance = df[self.config.distance_col]
        if (distance <= 0).any():
            raise ValueError("rate_per_km transforms require positive distance values")
        return distance

    def _date_offsets(self, df: pd.DataFrame) -> np.ndarray:
        if self.date_origin_ is None:
            raise RuntimeError("date trend must be fitted before transformation")
        dates = pd.to_datetime(df[self.config.date_col], errors="raise")
        return (dates - self.date_origin_).dt.total_seconds().to_numpy() / 86_400

    def _fit_target_transform(self, df: pd.DataFrame) -> None:
        if self.config.target_transform != "log_rate_per_km_detrended":
            return

        distance = self._positive_distance(df)
        rate_per_km = df[self.config.target_col] / distance
        if (rate_per_km <= 0).any():
            raise ValueError("log_rate_per_km_detrended requires positive target values")

        dates = pd.to_datetime(df[self.config.date_col], errors="raise")
        self.date_origin_ = dates.min()
        day_offsets = self._date_offsets(df)
        log_rate_per_km = np.log(rate_per_km.to_numpy())
        if np.ptp(day_offsets) == 0:
            self.date_trend_slope_ = 0.0
            self.date_trend_intercept_ = float(log_rate_per_km.mean())
        else:
            slope, intercept = np.polyfit(day_offsets, log_rate_per_km, 1)
            self.date_trend_slope_ = float(slope)
            self.date_trend_intercept_ = float(intercept)

    def _date_trend(self, df: pd.DataFrame) -> np.ndarray:
        if self.date_trend_intercept_ is None or self.date_trend_slope_ is None:
            raise RuntimeError("date trend must be fitted before transformation")
        return self.date_trend_intercept_ + self.date_trend_slope_ * self._date_offsets(df)

    def _select_xy(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        X = df[self.config.feature_cols]
        y = df[self.config.target_col]
        if self.config.target_transform == "rate_per_km":
            distance = self._positive_distance(df)
            y = y / distance
        elif self.config.target_transform == "log_rate_per_km_detrended":
            distance = self._positive_distance(df)
            y = pd.Series(
                np.log((y / distance).to_numpy()) - self._date_trend(df),
                index=y.index,
                name=y.name,
            )
        elif self.config.target_transform != "direct":
            raise ValueError(
                f"unknown target transform: {self.config.target_transform}"
            )
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
        self._fit_target_transform(df)
        X, y = self._select_xy(df)
        sample_weight = self._select_weights(df)

        fit_kwargs: dict[str, Any] = {}
        fit_params = inspect.signature(self.model.fit).parameters

        if self.config.categorical_cols and "cat_features" in fit_params:
            fit_kwargs["cat_features"] = [
                c for c in self.config.categorical_cols if c in X.columns
            ]

        if eval_df is not None and "eval_set" in fit_params:
            X_val, y_val = self._select_xy(eval_df)
            is_catboost = type(self.model).__module__.startswith("catboost")
            if is_catboost:
                # CatBoost auto-tracks a "learn" curve from the X/y already
                # passed to fit; giving it the training pair again as part
                # of eval_set would add a spurious extra key that shadows
                # the real validation curve in evals_result_.
                fit_kwargs["eval_set"] = [(X_val, y_val)]
            else:
                fit_kwargs["eval_set"] = [(X, y), (X_val, y_val)]
                if "eval_names" in fit_params:
                    fit_kwargs["eval_names"] = ["training", "valid"]

            if early_stopping_rounds is not None:
                if "early_stopping_rounds" in fit_params:
                    fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
                elif "callbacks" in fit_params:
                    import lightgbm as lgb

                    fit_kwargs["callbacks"] = [lgb.early_stopping(early_stopping_rounds)]

        self.model.fit(X, y, sample_weight=sample_weight, **fit_kwargs)
        self.is_fitted_ = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("model must be trained before predict")
        X = df[self.config.feature_cols]
        predictions = self.model.predict(X)
        if self.config.target_transform == "rate_per_km":
            predictions = predictions * df[self.config.distance_col].to_numpy()
        elif self.config.target_transform == "log_rate_per_km_detrended":
            predictions = (
                np.exp(predictions + self._date_trend(df))
                * df[self.config.distance_col].to_numpy()
            )
        return predictions

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
