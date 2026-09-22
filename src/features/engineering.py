"""Feature engineering pipeline for the freight rate dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.cleaning import BaseCleaner


class LaneBuilder(BaseCleaner):
    """Builds a lane identifier from pickup and delivery."""

    def __init__(
        self,
        pickup_col: str = "pickup",
        delivery_col: str = "delivery",
        out_col: str = "lane",
    ) -> None:
        super().__init__()
        self.pickup_col = pickup_col
        self.delivery_col = delivery_col
        self.out_col = out_col

    def fit(self, df: pd.DataFrame) -> LaneBuilder:
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        self._check_fitted()
        df = df.copy()
        df[self.out_col] = df[self.pickup_col] + " -> " + df[self.delivery_col]
        return df


class CyclicalMonthEncoder(BaseCleaner):
    """Encodes month as sin/cos pairs."""

    def __init__(self, date_col: str = "date") -> None:
        super().__init__()
        self.date_col = date_col

    def fit(self, df: pd.DataFrame) -> CyclicalMonthEncoder:
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        self._check_fitted()
        df = df.copy()
        month = pd.to_datetime(df[self.date_col]).dt.month
        df["month_sin"] = np.sin(2 * np.pi * month / 12)
        df["month_cos"] = np.cos(2 * np.pi * month / 12)
        return df


class DayOfWeekEncoder(BaseCleaner):
    """Extracts day of week as an integer, 0 through 6."""

    def __init__(self, date_col: str = "date", out_col: str = "day_of_week") -> None:
        super().__init__()
        self.date_col = date_col
        self.out_col = out_col

    def fit(self, df: pd.DataFrame) -> DayOfWeekEncoder:
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        self._check_fitted()
        df = df.copy()
        df[self.out_col] = pd.to_datetime(df[self.date_col]).dt.dayofweek
        return df


class EquipmentEncoder(BaseCleaner):
    """One-hot encodes equipment using categories learned from training."""

    def __init__(self, equipment_col: str = "equipment") -> None:
        super().__init__()
        self.equipment_col = equipment_col
        self.categories_: list[str] = []

    def fit(self, df: pd.DataFrame) -> EquipmentEncoder:
        self.categories_ = sorted(df[self.equipment_col].dropna().unique().tolist())
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        self._check_fitted()
        df = df.copy()
        for cat in self.categories_:
            df[f"equipment_{cat}"] = (df[self.equipment_col] == cat).astype(int)
        return df


class MarketIndexEMA(BaseCleaner):
    """Computes a per-lane exponential moving average of market_index.

    On training data, state updates chronologically row-by-row since every
    row belongs to the fold's training set. On non-training data
    (is_training=False), each row is blended against the frozen
    lane_last_ema_ learned in fit() only: ema = alpha * current_value +
    (1 - alpha) * fitted_training_lane_ema. Validation/holdout rows never
    update the running state or influence each other, so a row's output
    depends only on its own value and fitted training state, never on
    which other rows are in the same batch or their order.
    """

    def __init__(
        self,
        lane_col: str = "lane",
        date_col: str = "date",
        value_col: str = "market_index",
        span: int = 7,
        out_col: str = "market_index_ema",
    ) -> None:
        super().__init__()
        self.lane_col = lane_col
        self.date_col = date_col
        self.value_col = value_col
        self.span = span
        self.out_col = out_col
        self.alpha_ = 2 / (span + 1)
        self.lane_last_ema_: dict[str, float] = {}
        self.global_fallback_: float = np.nan

    def fit(self, df: pd.DataFrame) -> MarketIndexEMA:
        ordered = df.sort_values(self.date_col)
        self.global_fallback_ = ordered[self.value_col].mean()

        last_ema: dict[str, float] = {}
        for lane, group in ordered.groupby(self.lane_col):
            ema = None
            for value in group[self.value_col]:
                ema = (
                    value
                    if ema is None
                    else self.alpha_ * value + (1 - self.alpha_) * ema
                )
            if ema is not None:
                last_ema[lane] = ema

        self.lane_last_ema_ = last_ema
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        self._check_fitted()
        df = df.copy()

        if not is_training:
            fitted = df[self.lane_col].map(self.lane_last_ema_)
            fitted = fitted.fillna(self.global_fallback_)
            values = df[self.value_col]
            ema = self.alpha_ * values + (1 - self.alpha_) * fitted
            ema = ema.where(fitted.notna(), values)
            df[self.out_col] = ema
            return df

        ordered = df.sort_values(self.date_col)
        state: dict[str, float] = {}
        ema_values = pd.Series(index=ordered.index, dtype=float)

        for idx, row in ordered.iterrows():
            lane = row[self.lane_col]
            value = row[self.value_col]
            prev = state.get(lane, self.global_fallback_)
            ema = (
                value
                if pd.isna(prev)
                else self.alpha_ * value + (1 - self.alpha_) * prev
            )
            state[lane] = ema
            ema_values.loc[idx] = ema

        df[self.out_col] = ema_values.reindex(df.index)
        return df


class QuoteSignalZScore(BaseCleaner):
    """Computes a z-score for quote_signal using training mean and std."""

    def __init__(
        self, value_col: str = "quote_signal", out_col: str = "quote_signal_zscore"
    ) -> None:
        super().__init__()
        self.value_col = value_col
        self.out_col = out_col
        self.mean_: float = np.nan
        self.std_: float = np.nan

    def fit(self, df: pd.DataFrame) -> QuoteSignalZScore:
        self.mean_ = df[self.value_col].mean()
        self.std_ = df[self.value_col].std()
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        self._check_fitted()
        df = df.copy()
        df[self.out_col] = (df[self.value_col] - self.mean_) / self.std_
        return df


class FeatureEngineeringPipeline:
    """Runs a list of feature steps in order."""

    def __init__(self, steps: list[BaseCleaner]) -> None:
        self.steps = steps

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        current = df
        for step in self.steps:
            step.fit(current)
            current = step.transform(current, is_training=True)
        return current

    def transform(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        current = df
        for step in self.steps:
            current = step.transform(current, is_training=is_training)
        return current

    def save(self, path: str) -> None:
        import joblib

        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> FeatureEngineeringPipeline:
        import joblib

        return joblib.load(path)


FAMILY_STEPS: dict[str, list[type[BaseCleaner]]] = {
    "calendar": [CyclicalMonthEncoder, DayOfWeekEncoder],
    "lane": [LaneBuilder],
    "equipment": [EquipmentEncoder],
    "market_index_ema": [MarketIndexEMA],
    "quote_signal_zscore": [QuoteSignalZScore],
}

# Families whose steps depend on the lane column that LaneBuilder produces.
_REQUIRES_LANE = {"lane", "market_index_ema"}


def build_default_feature_pipeline(
    families: list[str] | None = None,
) -> FeatureEngineeringPipeline:
    """Builds a feature pipeline from the requested family names.

    families should match the keys in configs/features.yaml. LaneBuilder
    runs automatically, once, whenever any requested family needs it (lane
    itself, or market_index_ema, which depends on the lane column), even if
    "lane" was not explicitly requested.
    """
    if families is None:
        families = list(FAMILY_STEPS.keys())

    unknown = [f for f in families if f not in FAMILY_STEPS]
    if unknown:
        raise ValueError(f"unknown feature families: {unknown}")

    needs_lane = any(f in _REQUIRES_LANE for f in families)

    steps: list[BaseCleaner] = []
    if needs_lane:
        steps.append(LaneBuilder())

    for family in families:
        if family == "lane":
            continue
        for step_cls in FAMILY_STEPS[family]:
            steps.append(step_cls())

    return FeatureEngineeringPipeline(steps)
