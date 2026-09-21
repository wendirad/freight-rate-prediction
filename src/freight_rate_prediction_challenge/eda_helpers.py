"""Shared data loading and analysis helpers for the EDA notebooks."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path

import marimo as mo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.mixture import GaussianMixture

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class EDAConfig:
    """Dataset paths and column groups shared by the EDA notebooks."""

    train_test_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data/raw/train-test.csv"
    )
    target: str = "posted_rate"
    id_col: str = "load_id"
    categorical_cols: tuple[str, ...] = ("pickup", "delivery", "equipment")
    numerical_cols: tuple[str, ...] = (
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
        "distance",
        "weight",
        "market_index",
        "quote_signal",
    )
    date_cols: tuple[str, ...] = ("date",)
    total_cols: int = 14

    def __post_init__(self) -> None:
        if not self.train_test_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.train_test_path}")


def load_csv(path: str | Path, date_cols: Sequence[str] | None = None) -> pd.DataFrame:
    """Load a CSV file, optionally parsing selected date columns."""

    return pd.read_csv(Path(path), parse_dates=list(date_cols or ()))


def load_freight_data(config: EDAConfig | None = None) -> pd.DataFrame:
    """Load the freight dataset and validate its expected width."""

    cfg = config or EDAConfig()
    data = load_csv(cfg.train_test_path, cfg.date_cols)
    if data.shape[1] != cfg.total_cols:
        raise ValueError(f"Expected {cfg.total_cols} columns, found {data.shape[1]}")
    return data


def plot_qq(
    values: Sequence[Real],
    figsize: tuple[float, float] = (4, 3),
    dpi: int = 80,
) -> plt.Figure:
    """Return a Q-Q plot comparing numeric values with a normal distribution."""

    figure, axis = plt.subplots(figsize=figsize, dpi=dpi)
    stats.probplot(pd.Series(values).dropna(), dist="norm", plot=axis)
    figure.tight_layout()
    return figure


def within_std_proportions(values: Sequence[Real], n: int = 3) -> pd.DataFrame:
    """Compare empirical and normal proportions within one through ``n`` SDs."""

    clean_values = pd.Series(values, dtype="float64").dropna()
    z_scores = stats.zscore(clean_values)
    empirical = np.array([(np.abs(z_scores) <= k).mean() for k in range(1, n + 1)])
    theoretical = np.array(
        [stats.norm.cdf(k) - stats.norm.cdf(-k) for k in range(1, n + 1)]
    )
    return pd.DataFrame(
        {
            "Standard Deviations (k)": [f"±{k} SD" for k in range(1, n + 1)],
            "Empirical Proportion": empirical,
            "Offset": empirical - theoretical,
        }
    )


def report_findings(
    value: str,
    title: str = "Findings",
    kind: str = "info",
) -> mo.Html:
    """Wrap markdown findings in a consistently styled Marimo callout."""

    return mo.callout(title=title, kind=kind, value=value)


def cramers_v(confusion_matrix: pd.DataFrame) -> float:
    """Calculate Cramér's V for a contingency table."""

    chi_squared = chi2_contingency(confusion_matrix)[0]
    observations = confusion_matrix.to_numpy().sum()
    rows, columns = confusion_matrix.shape
    denominator = observations * (min(rows, columns) - 1)
    return float(np.sqrt(chi_squared / denominator)) if denominator else float("nan")


def haversine(
    lat1: float | np.ndarray | pd.Series,
    lon1: float | np.ndarray | pd.Series,
    lat2: float | np.ndarray | pd.Series,
    lon2: float | np.ndarray | pd.Series,
) -> float | np.ndarray | pd.Series:
    """Calculate great-circle distance in kilometres for scalars or arrays."""

    earth_radius_km = 6371
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, (lat1, lon1, lat2, lon2))
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    haversine_term = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    )
    return 2 * earth_radius_km * np.arcsin(np.sqrt(haversine_term))


def find_cluster_driver(
    data: pd.DataFrame,
    value_col: str,
    candidates: Sequence[str],
    n_components: int = 3,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank categorical columns by association with numeric mixture clusters."""

    values = data[value_col].dropna()
    mixture = GaussianMixture(n_components=n_components, random_state=random_state)
    clustered = data.loc[values.index].copy()
    clustered["_cluster"] = mixture.fit_predict(values.to_numpy().reshape(-1, 1))

    results: list[dict[str, str | float | int]] = []
    for column in candidates:
        if column not in clustered.columns:
            continue
        subset = clustered[[column, "_cluster"]].dropna()
        table = pd.crosstab(subset[column], subset["_cluster"])
        if table.shape[0] < 2:
            continue
        results.append(
            {
                "column": column,
                "cramers_v": cramers_v(table),
                "n_categories": table.shape[0],
            }
        )

    result_frame = pd.DataFrame(
        results, columns=["column", "cramers_v", "n_categories"]
    )
    return result_frame.sort_values("cramers_v", ascending=False), clustered


def recover_market_index(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Fill market index from the median of identical lane/date observations."""

    recovered = data.copy()
    lookup = (
        recovered.dropna(subset=["market_index"])
        .groupby(["pickup", "delivery", "date"])["market_index"]
        .median()
    )
    missing_mask = recovered["market_index"].isna()
    missing_keys = pd.MultiIndex.from_frame(
        recovered.loc[missing_mask, ["pickup", "delivery", "date"]]
    )
    recovered_values = pd.Series(
        lookup.reindex(missing_keys).to_numpy(),
        index=recovered.index[missing_mask],
        name="recovered_market_index",
    )
    recovered.loc[recovered_values.dropna().index, "market_index"] = (
        recovered_values.dropna()
    )
    return recovered, recovered_values
