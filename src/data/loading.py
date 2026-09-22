"""
Data loading for the freight rate dataset.

Responsible for reading the raw file, validating that the expected schema
is present before anything downstream touches it, enforcing chronological
order, and producing the train, validation, and test splits by date cutoff.
Cleaning and feature engineering are deliberately kept out of this module,
loading's only job is to get a trustworthy, correctly ordered dataframe
into memory and split it, everything else happens in cleaning.py and
features/engineering.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class SchemaConfig:
    """Expected columns and their raw dtypes, checked at load time."""

    required_columns: list[str] = field(
        default_factory=lambda: [
            "load_id",
            "pickup",
            "delivery",
            "pickup_lat",
            "pickup_lon",
            "delivery_lat",
            "delivery_lon",
            "distance",
            "equipment",
            "weight",
            "date",
            "market_index",
            "quote_signal",
            "posted_rate",
        ]
    )
    date_col: str = "date"


class SchemaValidationError(Exception):
    pass


class DataLoader:
    """Loads the raw dataset, validates its schema, and produces
    chronologically ordered train, validation, and test splits.

    Usage:
        loader = DataLoader("data/raw/train_test.csv")
        df = loader.load()
        train, val, test = loader.split(df, val_start="2025-09-01", test_start="2025-10-01")
    """

    def __init__(self, path: str | Path, schema: SchemaConfig | None = None) -> None:
        self.path = Path(path)
        self.schema = schema or SchemaConfig()
        self.load_report_: dict = {}

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"no file found at {self.path}")

        df = pd.read_csv(self.path)
        self._validate_schema(df)

        df[self.schema.date_col] = pd.to_datetime(df[self.schema.date_col])
        df = df.sort_values(self.schema.date_col).reset_index(drop=True)

        self.load_report_ = {
            "rows": len(df),
            "columns": list(df.columns),
            "date_min": df[self.schema.date_col].min(),
            "date_max": df[self.schema.date_col].max(),
        }
        return df

    def _validate_schema(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.schema.required_columns if c not in df.columns]
        if missing:
            raise SchemaValidationError(f"missing expected columns: {missing}")

    def split(
        self,
        df: pd.DataFrame,
        val_start: str,
        test_start: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Chronological split. Rows before val_start go to train, rows
        from val_start up to (not including) test_start go to validation,
        rows from test_start onward go to test. If test_start is None,
        everything from val_start onward is validation and test is empty.
        """
        date_col = self.schema.date_col
        val_start_ts = pd.Timestamp(val_start)

        train = df[df[date_col] < val_start_ts].reset_index(drop=True)

        if test_start is None:
            val = df[df[date_col] >= val_start_ts].reset_index(drop=True)
            test = df.iloc[0:0].reset_index(drop=True)
        else:
            test_start_ts = pd.Timestamp(test_start)
            val = df[
                (df[date_col] >= val_start_ts) & (df[date_col] < test_start_ts)
            ].reset_index(drop=True)
            test = df[df[date_col] >= test_start_ts].reset_index(drop=True)

        self._check_split_sanity(train, val, test, date_col)
        return train, val, test

    def _check_split_sanity(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        date_col: str,
    ) -> None:
        if len(train) == 0:
            raise ValueError("train split is empty, check val_start")
        if len(val) == 0:
            raise ValueError("validation split is empty, check val_start/test_start")
        if not train[date_col].max() <= val[date_col].min():
            raise ValueError("train and validation splits overlap in time")
        if len(test) > 0 and not val[date_col].max() <= test[date_col].min():
            raise ValueError("validation and test splits overlap in time")


def load_and_split(
    path: str | Path,
    val_start: str,
    test_start: str | None = None,
    schema: SchemaConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience function wrapping DataLoader for the common case."""
    loader = DataLoader(path, schema=schema)
    df = loader.load()
    return loader.split(df, val_start=val_start, test_start=test_start)
