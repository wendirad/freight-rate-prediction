"""
Cleaning pipeline for the freight rate dataset.

Design follows the sklearn fit/transform contract so every cleaner learns
state from training data only, then applies that learned state consistently
to validation, test, or live prediction data. Row-dropping steps (outlier
removal, duplicate removal) only ever run when is_training=True, since the
scorer requires a prediction for every row in validation and test, corrupted
or not. Imputation and normalization run identically on both paths.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class BaseCleaner(ABC):
    """Base contract for every cleaning step.

    fit learns whatever state is needed from training data (medians, lane
    lookups, thresholds). transform applies that learned state to any
    dataframe. is_training gates steps that are only valid to run on the
    data the model learns from, never on data the model must score.
    """

    def __init__(self) -> None:
        self._is_fitted = False

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> BaseCleaner: ...

    @abstractmethod
    def transform(
        self, df: pd.DataFrame, is_training: bool = False
    ) -> pd.DataFrame: ...

    def fit_transform(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df, is_training=is_training)

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fit before transform"
            )


class TypeCaster(BaseCleaner):
    """Parses date into datetime and casts numeric columns to consistent dtypes."""

    def __init__(
        self, date_col: str = "date", numeric_cols: list[str] | None = None
    ) -> None:
        super().__init__()
        self.date_col = date_col
        self.numeric_cols = numeric_cols or [
            "pickup_lat",
            "pickup_lon",
            "delivery_lat",
            "delivery_lon",
            "distance",
            "weight",
            "market_index",
            "quote_signal",
            "posted_rate",
        ]

    def fit(self, df: pd.DataFrame) -> TypeCaster:
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        self._check_fitted()
        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        for col in self.numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df


class CategoryNormalizer(BaseCleaner):
    """Strips whitespace and normalizes casing for categorical location columns."""

    def __init__(self, columns: list[str] | None = None) -> None:
        super().__init__()
        self.columns = columns or ["pickup", "delivery", "equipment"]

    def fit(self, df: pd.DataFrame) -> CategoryNormalizer:
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        self._check_fitted()
        df = df.copy()
        for col in self.columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        return df


class WeightCleaner(BaseCleaner):
    """Repairs sign-flipped negative weights and imputes missing weight.

    Negative weights are converted to their absolute value rather than
    dropped, since their magnitude sits within the expected weight range,
    a sign flip is a more defensible explanation than a corrupted value.
    Missing weight is filled with the training median weight for that
    load's equipment type, falling back to the overall training median if
    an equipment group has no valid weight rows to learn from. This must
    run on both training and prediction data, since a missing weight at
    prediction time cannot be dropped, every load needs a rate.
    """

    def __init__(
        self, equipment_col: str = "equipment", weight_col: str = "weight"
    ) -> None:
        super().__init__()
        self.equipment_col = equipment_col
        self.weight_col = weight_col
        self.equipment_medians_: dict[str, float] = {}
        self.global_median_: float = np.nan

    def fit(self, df: pd.DataFrame) -> WeightCleaner:
        valid = df[df[self.weight_col].abs() > 0].copy()
        valid[self.weight_col] = valid[self.weight_col].abs()
        self.equipment_medians_ = (
            valid.groupby(self.equipment_col)[self.weight_col].median().to_dict()
        )
        self.global_median_ = valid[self.weight_col].median()
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        self._check_fitted()
        df = df.copy()

        df[self.weight_col] = df[self.weight_col].abs()

        missing_mask = df[self.weight_col].isna()
        if missing_mask.any():
            fallback = df.loc[missing_mask, self.equipment_col].map(
                self.equipment_medians_
            )
            fallback = fallback.fillna(self.global_median_)
            df.loc[missing_mask, self.weight_col] = fallback

        return df


class MarketIndexImputer(BaseCleaner):
    """Fills missing market_index in three stages, most trustworthy first.

    Stage one matches a missing row against any other row in the same
    batch sharing pickup, delivery, and date, using the median of those
    matches, since that reflects the same lane under the same day's
    market conditions. Stage two falls back to a lane level median
    learned from training (pickup, delivery). Stage three falls back to
    a month level median learned from training, respecting the seasonal
    structure confirmed during EDA. A global training median is the last
    resort. Runs on both training and prediction data.
    """

    def __init__(
        self,
        pickup_col: str = "pickup",
        delivery_col: str = "delivery",
        date_col: str = "date",
        target_col: str = "market_index",
    ) -> None:
        super().__init__()
        self.pickup_col = pickup_col
        self.delivery_col = delivery_col
        self.date_col = date_col
        self.target_col = target_col
        self.lane_medians_: dict[tuple, float] = {}
        self.month_medians_: dict[int, float] = {}
        self.global_median_: float = np.nan

    def fit(self, df: pd.DataFrame) -> MarketIndexImputer:
        known = df[df[self.target_col].notna()].copy()

        self.lane_medians_ = (
            known.groupby([self.pickup_col, self.delivery_col])[self.target_col]
            .median()
            .to_dict()
        )

        months = pd.to_datetime(known[self.date_col]).dt.month
        self.month_medians_ = known.groupby(months)[self.target_col].median().to_dict()

        self.global_median_ = known[self.target_col].median()
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        self._check_fitted()
        df = df.copy()

        missing_mask = df[self.target_col].isna()
        if not missing_mask.any():
            return df

        known = df[df[self.target_col].notna()][
            [self.pickup_col, self.delivery_col, self.date_col, self.target_col]
        ]
        lane_date_median = known.groupby(
            [self.pickup_col, self.delivery_col, self.date_col]
        )[self.target_col].median()

        missing_idx = df.index[missing_mask]
        keys = list(
            zip(
                df.loc[missing_idx, self.pickup_col],
                df.loc[missing_idx, self.delivery_col],
                df.loc[missing_idx, self.date_col],
            )
        )
        stage_one = pd.Series(
            [lane_date_median.get(k, np.nan) for k in keys], index=missing_idx
        )
        df.loc[missing_idx, self.target_col] = stage_one

        still_missing = df[self.target_col].isna()
        if still_missing.any():
            lane_keys = list(
                zip(
                    df.loc[still_missing, self.pickup_col],
                    df.loc[still_missing, self.delivery_col],
                )
            )
            stage_two = pd.Series(
                [self.lane_medians_.get(k, np.nan) for k in lane_keys],
                index=df.index[still_missing],
            )
            df.loc[still_missing, self.target_col] = stage_two

        still_missing = df[self.target_col].isna()
        if still_missing.any():
            months = pd.to_datetime(df.loc[still_missing, self.date_col]).dt.month
            stage_three = months.map(self.month_medians_)
            df.loc[still_missing, self.target_col] = stage_three.values

        df[self.target_col] = df[self.target_col].fillna(self.global_median_)

        return df


class DuplicateRemover(BaseCleaner):
    """Drops exact duplicate rows and duplicate load_id, training only.

    Runs its check on every call and logs what it finds regardless of
    is_training, but only actually removes rows when is_training=True,
    since prediction data must return a row for every load_id it receives.
    """

    def __init__(self, id_col: str = "load_id") -> None:
        super().__init__()
        self.id_col = id_col
        self.last_report_: dict = {}

    def fit(self, df: pd.DataFrame) -> DuplicateRemover:
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        self._check_fitted()
        df = df.copy()

        exact_dupes = df.duplicated().sum()
        id_dupes = (
            df[self.id_col].duplicated().sum() if self.id_col in df.columns else 0
        )
        self.last_report_ = {
            "exact_duplicates": int(exact_dupes),
            "duplicate_ids": int(id_dupes),
        }

        if is_training:
            df = df.drop_duplicates()
            if self.id_col in df.columns:
                df = df.drop_duplicates(subset=[self.id_col])

        return df


class TargetOutlierRemover(BaseCleaner):
    """Removes extreme posted_rate outliers, training only.

    Bounds are learned from training data quantiles and never recomputed
    on later data, avoiding leakage. Never runs on non-training data,
    since dropping rows the model finds hard to predict would artificially
    inflate validation and test performance rather than reflect it.
    """

    def __init__(
        self,
        target_col: str = "posted_rate",
        lower_q: float = 0.005,
        upper_q: float = 0.995,
    ) -> None:
        super().__init__()
        self.target_col = target_col
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.lower_bound_: float = np.nan
        self.upper_bound_: float = np.nan

    def fit(self, df: pd.DataFrame) -> TargetOutlierRemover:
        self.lower_bound_ = df[self.target_col].quantile(self.lower_q)
        self.upper_bound_ = df[self.target_col].quantile(self.upper_q)
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        self._check_fitted()
        df = df.copy()
        if not is_training:
            return df
        mask = (df[self.target_col] >= self.lower_bound_) & (
            df[self.target_col] <= self.upper_bound_
        )
        return df[mask]


class LaneDistanceOutlierRemover(BaseCleaner):
    """Drops rows where distance deviates sharply from its lane's usual distance, training only.

    A lane's expected distance is learned as the median distance for that
    pickup/delivery pair in training data. Rows more than the configured
    tolerance away from that lane median are dropped, since a lane with a
    stable median distance producing a wildly different value for one row
    suggests a data entry error rather than genuine variation.
    """

    def __init__(
        self,
        pickup_col: str = "pickup",
        delivery_col: str = "delivery",
        distance_col: str = "distance",
        tolerance: float = 0.20,
    ) -> None:
        super().__init__()
        self.pickup_col = pickup_col
        self.delivery_col = delivery_col
        self.distance_col = distance_col
        self.tolerance = tolerance
        self.lane_medians_: dict[tuple, float] = {}

    def fit(self, df: pd.DataFrame) -> LaneDistanceOutlierRemover:
        self.lane_medians_ = (
            df.groupby([self.pickup_col, self.delivery_col])[self.distance_col]
            .median()
            .to_dict()
        )
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        self._check_fitted()
        df = df.copy()
        if not is_training:
            return df

        keys = list(zip(df[self.pickup_col], df[self.delivery_col]))
        expected = pd.Series(
            [self.lane_medians_.get(k, np.nan) for k in keys], index=df.index
        )
        deviation = (df[self.distance_col] - expected).abs() / expected
        mask = deviation.isna() | (deviation <= self.tolerance)
        return df[mask]


@dataclass
class CleaningReport:
    step: str
    rows_before: int
    rows_after: int
    extra: dict = field(default_factory=dict)


class CleaningPipeline:
    """Runs a list of cleaners in order, fitting on training data and
    applying consistently to any other split via transform.

    fit_transform is used once on the training split. transform (without
    fit) is used on validation, test, or live prediction data, reusing
    everything learned from training, with is_training=False so that
    outlier and duplicate removal steps are skipped and every row is
    preserved for scoring.
    """

    def __init__(self, steps: list[BaseCleaner]) -> None:
        self.steps = steps
        self.reports_: list[CleaningReport] = []

    def fit(self, df: pd.DataFrame) -> CleaningPipeline:
        current = df
        for step in self.steps:
            step.fit(current)
            current = step.transform(current, is_training=True)
        return self

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        current = df
        self.reports_ = []
        for step in self.steps:
            before = len(current)
            step.fit(current)
            current = step.transform(current, is_training=True)
            self.reports_.append(
                CleaningReport(
                    step=step.__class__.__name__,
                    rows_before=before,
                    rows_after=len(current),
                )
            )
        return current

    def transform(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        current = df
        self.reports_ = []
        for step in self.steps:
            before = len(current)
            current = step.transform(current, is_training=is_training)
            self.reports_.append(
                CleaningReport(
                    step=step.__class__.__name__,
                    rows_before=before,
                    rows_after=len(current),
                )
            )
        return current

    def save(self, path: str) -> None:
        import joblib

        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> CleaningPipeline:
        import joblib

        return joblib.load(path)


def build_default_pipeline() -> CleaningPipeline:
    """Assembles the cleaning pipeline matching the EDA findings and the
    training-only versus always split used across similar freight rate
    prediction pipelines.
    """
    return CleaningPipeline(
        [
            TypeCaster(),
            CategoryNormalizer(),
            WeightCleaner(),
            MarketIndexImputer(),
            DuplicateRemover(),
            LaneDistanceOutlierRemover(),
            TargetOutlierRemover(),
        ]
    )
