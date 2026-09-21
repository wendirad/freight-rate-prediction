import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", layout_file="layouts/02_eda.slides.json")

with app.setup:
    import warnings
    from collections.abc import Sequence
    from dataclasses import dataclass
    from numbers import Real
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from geopy.extra.rate_limiter import RateLimiter
    from geopy.geocoders import Nominatim
    from scipy import stats
    from scipy.stats import chi2_contingency, linregress
    from sklearn.mixture import GaussianMixture

    warnings.filterwarnings("ignore")
    alt.data_transformers.enable("vegafusion")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # **Freight Rate Prediction**
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ### **Fields Description**

    | Column                          | Description                                                        |
    | ------------------------------- | ------------------------------------------------------------------ |
    | `load_id`                       | Random ID representing a single load, given at prediction time     |
    | `pickup`                        | Pickup location name                                               |
    | `delivery`                      | Delivery location name                                             |
    | `pickup_lat` / `pickup_lon`     | Pickup location coordinates                                        |
    | `delivery_lat` / `delivery_lon` | Delivery location coordinates                                      |
    | `distance`                      | Distance between pickup and delivery, in kilometers                |
    | `equipment`                     | Carrier equipment type                                             |
    | `weight`                        | Load weight, in pounds (lbs)                                       |
    | `date`                          | Departure/arrival date for the lane                                |
    | `market_index`                  | Weighted average cost of supply and demand in the logistics market |
    | `quote_signal`                  | Live indicator of the real-time truck/cargo balance in the market  |
    | `posted_rate`                   | Target: total flat rate for the shipment                           |
    """)
    return


@app.cell
def _():
    @dataclass(frozen=True)
    class Config:
        # Path
        train_test_path: Path = Path("data/raw/train-test.csv")

        # Columns
        TARGET: str = "posted_rate"
        ID_COL: str = "load_id"
        CATEGORICAL_COLS: tuple[str] = ("pickup", "delivery", "equipment")
        NUMERICAL_COLS: tuple[str] = (
            "pickup_lat",
            "pickup_lon",
            "delivery_lat",
            "delivery_lon",
            "distance",
            "weight",
            "market_index",
            "quote_signal",
        )

        DATE_COLS: tuple[str] = ("date",)
        TOTAL_COLS: int = 14

        def __init__(self):
            assert self.train_test_path.exists(), (
                "The file or directories does not exist."
            )

    cfg = Config()
    return (cfg,)


@app.cell
def _():
    def load_csv(
        name: str | Path, date_cols: list[str] | None = None
    ) -> pd.DataFrame:
        """Loads a CSV file into a Pandas DataFrame."""

        filepath = Path(name) if isinstance(name, str) else name
        return pd.read_csv(filepath, parse_dates=date_cols)

    def plot_qq(
        values: Sequence[Real],
        figsize: tuple[float, float] = (4, 3),
        dpi: int = 80,
    ) -> None:
        """Draw a Q-Q plot comparing values against a normal distribution.

        values: the numeric data to test for normality
        figsize: width and height of the figure in inches
        dpi: resolution of the figure
        """
        plt.figure(figsize=figsize, dpi=dpi)
        stats.probplot(values, dist="norm", plot=plt)
        plt.show()

    def within_std_proportions(values: Sequence[Real], n: int) -> np.ndarray:
        """Compute the proportion of values within k standard deviations, for k = 1..n.

        values: the numeric data to standardize and check
        n: the number of standard deviation bands to check (e.g. n=2 checks 1sd and 2sd)

        Returns an array where index i holds the proportion within (i+1) standard deviations.
        Compare against the normal distribution reference: ~0.68 for 1sd, ~0.95 for 2sd, ~0.997 for 3sd.
        """
        # z = stats.zscore(values)
        # return np.array([(np.abs(z) <= k).mean() for k in range(1, n + 1)])

        z = stats.zscore(values)
        empirical_props = [(np.abs(z) <= k).mean() for k in range(1, n + 1)]

        # 2. Calculate theoretical Normal reference values for each k
        # stats.norm.cdf(k) - stats.norm.cdf(-k) gives the exact theoretical percentage
        theoretical_props = [
            stats.norm.cdf(k) - stats.norm.cdf(-k) for k in range(1, n + 1)
        ]

        # 3. Construct a clean DataFrame
        df = pd.DataFrame(
            {
                "Standard Deviations (k)": [
                    f"±{k} SD" for k in range(1, n + 1)
                ],
                "Empirical Proportion": empirical_props,
                "Offset": np.array(empirical_props)
                - np.array(theoretical_props),
            }
        )

        return df

    def report_findings(
        value: str,
        title: str = "Findings",
        kind: str = "info",
    ) -> mo.Html:
        """Wrap a findings message in a marimo callout.

        value: the callout body text (markdown supported)
        title: the callout heading
        kind: the callout style, e.g. "info", "warn", "danger", "success"
        """
        return mo.callout(title=title, kind=kind, value=value)

    def cramers_v(confusion_matrix: pd.DataFrame) -> float:
        """Compute Cramer's V, a measure of association between two categorical variables.

        confusion_matrix: a contingency table (crosstab) of the two variables

        Returns a value between 0 and 1, where 0 means no association and
        1 means a perfect association.
        """
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        r, k = confusion_matrix.shape
        return np.sqrt(chi2 / (n * (min(r, k) - 1)))

    def haversine(
        lat1: float | np.ndarray,
        lon1: float | np.ndarray,
        lat2: float | np.ndarray,
        lon2: float | np.ndarray,
    ) -> float | np.ndarray:
        """Compute the great circle distance between two points on Earth.

        lat1, lon1: latitude and longitude of the first point, in degrees
        lat2, lon2: latitude and longitude of the second point, in degrees

        Returns the distance in kilometers. Accepts scalars or arrays for
        vectorized computation over many point pairs at once.
        """
        R = 6371  # Earth radius in km

        lat1 = np.radians(lat1)
        lon1 = np.radians(lon1)
        lat2 = np.radians(lat2)
        lon2 = np.radians(lon2)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        )

        return 2 * R * np.arcsin(np.sqrt(a))

    def find_cluster_driver(
        df: pd.DataFrame,
        value_col: str,
        candidates: list[str],
        n_components: int = 3,
        random_state: int = 0,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Find which categorical column best explains a numeric column's multimodality.

        df: the source dataframe
        value_col: the numeric column to cluster, e.g. "market_index" or "distance"
        candidates: list of categorical column names to test for association with the clusters
        n_components: number of clusters to fit with the Gaussian mixture, matching the number of peaks seen
        random_state: seed for reproducible cluster assignment

        Returns a tuple of (results, clustered):
        results is a dataframe sorted by cramers_v descending, with a column
        for each candidate's association strength and category count.
        clustered is the original dataframe (rows with non-null value_col only)
        with a "_cluster" column added, for further analysis like cross tabs
        against another column's clusters.
        Columns with very high category counts relative to the row count
        can inflate cramers_v artificially, so treat those results with caution.
        """
        values = df[value_col].dropna()
        gmm = GaussianMixture(
            n_components=n_components, random_state=random_state
        )
        cluster_labels = gmm.fit_predict(values.values.reshape(-1, 1))

        clustered = df.loc[values.index].copy()
        clustered["_cluster"] = cluster_labels

        results = []
        for col in candidates:
            if col not in clustered.columns:
                continue
            sub = clustered[[col, "_cluster"]].dropna()
            table = pd.crosstab(sub[col], sub["_cluster"])
            if table.shape[0] < 2:
                continue
            results.append(
                {
                    "column": col,
                    "cramers_v": cramers_v(table),
                    "n_categories": table.shape[0],
                }
            )

        results_df = (
            pd.DataFrame(results)
            .sort_values("cramers_v", ascending=False)
            .reset_index(drop=True)
        )
        return results_df, clustered

    return (
        cramers_v,
        find_cluster_driver,
        haversine,
        load_csv,
        plot_qq,
        report_findings,
        within_std_proportions,
    )


@app.cell
def _(cfg, load_csv):
    def _():
        # Load Data
        df = load_csv(cfg.train_test_path, date_cols=list(cfg.DATE_COLS))
        assert df.shape[1] == cfg.TOTAL_COLS

        print("Shape:", df.shape)

        return df

    train_test = _()
    train_test_sample = train_test.sample(n=min(5000, len(train_test)), random_state=0)
    return train_test, train_test_sample


@app.cell
def _(train_test):
    with mo.redirect_stdout():
        train_test.info()
    return


@app.cell
def _(train_test):
    print("exact duplicate rows:", train_test.duplicated().sum())
    print("duplicate load_id:", train_test["load_id"].duplicated().sum())
    return


@app.cell
def _(train_test):
    def _():
        missing = train_test.isnull().sum()
        return missing[missing > 0].to_frame(name="Missing Count")

    _()
    return


@app.cell
def _(train_test):
    train_test.describe()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## **Explore Weight Destribution**
    """)
    return


@app.cell
def _(plot_qq, train_test):
    partial_weight = np.abs(train_test.weight[train_test.weight > 0]).dropna()
    plot_qq(partial_weight)
    return (partial_weight,)


@app.cell
def _(partial_weight, within_std_proportions):
    within_std_proportions(partial_weight, 3)
    return


@app.cell
def _(report_findings):
    report_findings(
        """After ignoring the invalid negative values, 67.18% of the weight observations fall within one standard deviation of the mean and 94.93% fall within two standard deviations. These are very close to the theoretical 68% and 95% expected for a normal distribution. The Q-Q plot supports this in the central portion of the data, where observations closely follow the reference line. However, deviations at both ends indicate that the distribution is not perfectly normal, mainly because weight has clear lower and upper boundaries, including a concentration near the maximum weight of 47,500."""
    )
    return


@app.cell
def _(train_test, train_test_sample):
    def _():
        sample = train_test.sample(n=min(5000, len(train_test)), random_state=0)

        base = (
            alt.Chart(train_test_sample)
            .mark_circle(opacity=0.5)
            .encode(
                x="distance:Q",
                y="weight:Q",
                color="equipment:N",
                tooltip=["distance", "weight", "equipment"],
            )
            .properties(width=660, height=300, title="Distance vs Weight by Equipment")
        )

        faceted = (
            alt.Chart(sample)
            .mark_circle(opacity=0.4)
            .encode(x="distance:Q", y="weight:Q")
            .properties(width=200, height=200)
            .facet(facet="equipment:N", columns=3)
        )

        return [base, faceted]

    mo.vstack(_())
    return


@app.cell(hide_code=True)
def _(report_findings):
    report_findings(
        """The combined analysis of distance and equipment type shows that neither variable strongly explains the variation in weight. Across Dry Van, Reefer, and Flatbed, weights remain widely distributed at nearly every distance, with substantial overlap between equipment types. There is no clear trend of weight increasing or decreasing with distance, and separating the data by equipment produces very similar patterns. This suggests that weight is likely driven by other shipment characteristics or interactions not captured by these two features alone."""
    )
    return


@app.cell
def _(partial_weight):
    print(partial_weight.skew())
    return


@app.cell
def _(train_test):
    sns.displot(
        data=train_test,
        x="weight",
        col="equipment",
        col_wrap=3,
        bins=40,
        kde=True,
        facet_kws={"sharey": False},
    )

    plt.show()
    return


@app.cell
def _(train_test):
    plt.figure(figsize=(4, 4))

    sns.boxplot(data=train_test, x="equipment", y="weight")

    plt.show()
    return


@app.cell
def _(train_test):
    train_test.groupby("equipment")["weight"].agg(
        ["count", "mean", "median", "std", "min", "max"]
    )
    return


@app.cell
def _(train_test):
    train_test.groupby("equipment")["weight"].describe(
        percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]
    )
    return


@app.cell(hide_code=True)
def _(report_findings):
    report_findings(
        """The weight distribution is nearly identical across Dry Van, Reefer, and Flatbed equipment. All three have a mean and median around 31K and a standard deviation of roughly 9K, with very similar ranges and percentiles. This suggests that equipment type alone does not have a meaningful effect on weight, and the variation in weight is likely explained by other features or combinations of features in the dataset."""
    )
    return


@app.cell
def _(train_test):
    train_test.assign(weight_missing=train_test.weight.isna()).groupby("equipment")[
        "weight_missing"
    ].mean()
    return


@app.cell
def _(report_findings):
    report_findings(
        """Removing negative weight observations had no meaningful effect on the weight distribution, with the proportions within one and two standard deviations remaining approximately 67.18% and 94.93%, respectively. Since missing weights represent only about 0.6% of the data and weight is also temporally dependent, attempting to recover or impute these values could introduce inaccurate assumptions by ignoring conditions specific to the date of each observation. Therefore, dropping these records was considered more appropriate than imputation.""",
        title="Conclusion",
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## **Market Index Analysis**
    """)
    return


@app.cell
def _(plot_qq, train_test):
    partial_market_index = train_test.market_index.dropna()
    plot_qq(partial_market_index)
    return (partial_market_index,)


@app.cell
def _(partial_weight, within_std_proportions):
    within_std_proportions(partial_weight, 3)
    return


@app.cell
def _(train_test):
    def _():
        missing = train_test[train_test["market_index"].isna()].copy()

        known = train_test[train_test["market_index"].notna()].copy()

        matches = missing[["load_id", "pickup", "delivery", "date"]].merge(
            known[["pickup", "delivery", "date", "market_index"]],
            on=["pickup", "delivery", "date"],
            how="left",
        )

        comparison = matches.groupby("load_id")["market_index"].agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            min="min",
            max="max",
        )

        return comparison["count"] > 0

    _recoverable = _()

    print("Recoverable:", _recoverable.sum())
    print("Recovery rate:", f"{round(_recoverable.mean() * 100, 2)}%")
    return


@app.cell
def _(report_findings):
    report_findings("""
    Among the 374 observations with missing market_index, only 24 could be matched with another observation having the same pickup, delivery, and date, giving a recovery rate of approximately 6.42%. Although this method provides limited coverage, these matches are the most suitable for recovery because they represent the same transportation lane under the same day's market conditions. The median market_index of matching observations was therefore used to recover these values.
    """)
    return


@app.cell
def _(partial_market_index):
    print(partial_market_index.skew())
    return


@app.cell
def _(train_test):
    sns.displot(
        data=train_test,
        x="market_index",
        kde=True,
        facet_kws={"sharey": False},
        height=3,
    )

    plt.show()
    return


@app.cell
def _(cfg):
    col_dropdown = mo.ui.dropdown(
        options=cfg.CATEGORICAL_COLS, value=cfg.CATEGORICAL_COLS[2]
    )
    col_dropdown
    return (col_dropdown,)


@app.cell
def _(col_dropdown, train_test):
    sns.displot(
        data=train_test,
        x="market_index",
        col=col_dropdown.value,
        col_wrap=3,
        bins=40,
        kde=True,
        facet_kws={"sharey": False},
        height=3,
    )

    plt.show()
    return


@app.cell
def _(train_test):
    plt.figure(figsize=(10, 1), dpi=80)

    sns.boxplot(data=train_test, x="market_index")

    plt.show()
    return


@app.cell
def _(report_findings):
    report_findings("""The market index distribution is multimodal, with three visible peaks around 1.0, 1.2, and 1.35, rather than a single unimodal shape, despite an overall skew of only 0.214 which masks this structure. The main mass sits between roughly 0.9 and 1.2, with a longer tail extending toward 1.45 to 1.5, and the boxplot shows no flagged outliers with whiskers spanning the full observed range.

    The market index's multimodal pattern persists identically across all three equipment types (Dry Van, Reefer, Flatbed), with only the counts scaling to match each category's sample size, ruling out equipment type as the source of the multimodality.""")
    return


@app.cell
def _(cfg, cramers_v, find_cluster_driver, train_test):
    col_candidates = cfg.CATEGORICAL_COLS + cfg.DATE_COLS


    _result, _clustered = find_cluster_driver(train_test, 'market_index', col_candidates)

    _clustered["month"] = pd.to_datetime(_clustered["date"]).dt.month
    _clustered["day_of_week"] = pd.to_datetime(_clustered["date"]).dt.dayofweek

    for _col in ["month", "day_of_week"]:
        _table = pd.crosstab(_clustered[_col], _clustered["_cluster"])
        print(_col, cramers_v(_table))

    _result
    return (col_candidates,)


@app.cell
def _(clustered, cramers_v):
    clustered["lane"] = clustered["pickup"] + " -> " + clustered["delivery"]
    _table = pd.crosstab(clustered["lane"], clustered["mi_cluster"])
    print("lane", cramers_v(_table))
    return


@app.cell
def _():
    _opts = ["month", "biweek", "triweek"]
    date_dropdown = mo.ui.dropdown(options=_opts, value=_opts[0])
    date_dropdown
    return (date_dropdown,)


@app.cell
def _(clustered, date_dropdown):
    clustered["month"] = pd.to_datetime(clustered["date"]).dt.month
    start = clustered["date"].min()
    days_elapsed = (clustered["date"] - start).dt.days

    clustered["biweek"] = days_elapsed // 14
    clustered["triweek"] = days_elapsed // 2

    g = sns.displot(
        data=clustered,
        x="market_index",
        col=date_dropdown.value,
        col_wrap=5,
        bins=40,
        kde=True,
        facet_kws={"sharey": False},
    )
    plt.show()
    return


@app.cell
def _(report_findings):
    report_findings(
        """The market_index's multimodal pattern holds up at every time resolution tested, monthly, biweekly, and triweekly, with the same peaks recurring inside each window rather than resolving into a single mode. This rules out calendar time granularity as the source of the multimodality."""
    )
    return


@app.cell
def _(train_test):
    train_test.assign(market_index_missing=train_test["market_index"].isna()).groupby(
        "equipment"
    )["market_index_missing"].mean()
    return


@app.cell
def _(report_findings):
    report_findings(
        """
    After recovering the reliable same-lane and same-date observations, the remaining missing values represent only a small proportion of the dataset. Missing rates are approximately 0.74% for Dry Van, 0.83% for Flatbed, and 0.82% for Reefer, showing similarly low missingness across equipment types. Since estimating the remaining values would require increasingly uncertain temporal assumptions, dropping these observations is expected to have substantially less impact on the dataset than introducing potentially inaccurate imputed values.""",
        title="Conclusion",
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## **Pickup and Delivery Analysis**
    """)
    return


@app.cell
def _(train_test):
    location_check = pd.concat(
        [
            train_test[["pickup", "pickup_lat", "pickup_lon"]].rename(
                columns={"pickup": "location", "pickup_lat": "lat", "pickup_lon": "lon"}
            ),
            train_test[["delivery", "delivery_lat", "delivery_lon"]].rename(
                columns={
                    "delivery": "location",
                    "delivery_lat": "lat",
                    "delivery_lon": "lon",
                }
            ),
        ]
    )

    print("Total unique location names:", location_check["location"].nunique())

    print(
        "Unique coordinate pairs:",
        location_check[["lat", "lon"]].drop_duplicates().shape[0],
    )

    print(location_check.isna().mean())
    return (location_check,)


@app.cell
def _(location_check):
    coord_per_location = location_check.groupby("location").agg(
        observations=("location", "size"),
        unique_lat=("lat", "nunique"),
        unique_lon=("lon", "nunique"),
    )

    coord_per_location["multiple_coordinates"] = (
        coord_per_location["unique_lat"] > 1
    ) | (coord_per_location["unique_lon"] > 1)

    print(coord_per_location.describe())

    print(coord_per_location["multiple_coordinates"].value_counts())

    print(
        coord_per_location[coord_per_location["multiple_coordinates"]]
        .sort_values("observations", ascending=False)
        .head(20)
    )
    return


@app.cell
def _():
    mo.callout(
        """The dataset contains 64 unique locations and 64 unique latitude/longitude pairs, with no missing geographic information. Each pickup or delivery location consistently maps to exactly one coordinate pair, with no locations associated with multiple coordinates. This indicates that the geographic variables are internally consistent and that the latitude/longitude columns provide a reliable numerical representation of the pickup and delivery locations.""",
        title="Findings",
        kind="info",
    )
    return


@app.cell
def _(haversine, train_test):
    df = train_test.copy()

    df["geo_distance"] = haversine(
        df["pickup_lat"], df["pickup_lon"], df["delivery_lat"], df["delivery_lon"]
    )

    df["distance_diff"] = df["distance"] - df["geo_distance"]

    print(df[["distance", "geo_distance", "distance_diff"]].describe())

    print("Correlation:", df["distance"].corr(df["geo_distance"]))
    return (df,)


@app.cell
def _(report_findings):
    report_findings(
        """When the straight-line geographic distance between pickup and delivery was calculated from the coordinates, it showed an extremely strong correlation of 0.9995 with the provided distance variable. This confirms that the coordinates and recorded distances represent highly consistent geographic relationships. However, the calculated geographic distance was, on average, approximately 419 km greater than the provided distance."""
    )
    return


@app.cell
def _(df):
    df["distance_ratio"] = df["distance"] / df["geo_distance"]

    print(df["distance_ratio"].describe())

    (
        df[["distance", "geo_distance", "distance_ratio"]]
        .sort_values("distance_ratio")
        .head(10)
    )
    return


@app.cell
def _(df):
    _result = linregress(df["geo_distance"], df["distance"])

    print("Slope:", _result.slope)
    print("Intercept:", _result.intercept)
    print("R²:", _result.rvalue**2)
    return


@app.cell
def _(report_findings):
    report_findings(
        """The provided distance has an almost perfect linear relationship with the coordinate-derived geographic distance (R² = 0.9991). The provided distances are typically around 73–74% of the calculated geographic distances, suggesting a systematic difference in scaling or calculation method or a random geographic inconsistencies."""
    )
    return


@app.cell
def _(df):
    def _():
        locations = sorted(set(df["pickup"].dropna()) | set(df["delivery"].dropna()))

        geolocator = Nominatim(user_agent="logistics_location_analysis")

        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

        results = []

        for city in locations:
            location = geocode(f"{city}, USA", exactly_one=True, country_codes="us")

            results.append(
                {
                    "location": city,
                    "lat": location.latitude if location else None,
                    "lon": location.longitude if location else None,
                    "matched_address": location.address if location else None,
                }
            )

        coords = pd.DataFrame(results)
        return coords

    fetched_coords = _()
    return (fetched_coords,)


@app.cell
def _(df, fetched_coords):
    current_coords = pd.concat(
        [
            df[["pickup", "pickup_lat", "pickup_lon"]].rename(
                columns={
                    "pickup": "location",
                    "pickup_lat": "current_lat",
                    "pickup_lon": "current_lon",
                }
            ),
            df[["delivery", "delivery_lat", "delivery_lon"]].rename(
                columns={
                    "delivery": "location",
                    "delivery_lat": "current_lat",
                    "delivery_lon": "current_lon",
                }
            ),
        ]
    ).drop_duplicates("location")

    coordinates = (
        current_coords.merge(fetched_coords, on="location", how="left")
        .rename(columns={"lat": "fetched_lat", "lon": "fetched_lon"})
        .drop("matched_address", axis=1)
    )

    coordinates["abs_lat_diff"] = abs(
        abs(coordinates["current_lat"]) - abs(coordinates["fetched_lat"])
    )
    coordinates["abs_lon_diff"] = abs(
        abs(coordinates["current_lon"]) - abs(coordinates["fetched_lon"])
    )

    coordinates
    return (coordinates,)


@app.cell
def _(coordinates):
    for _threshold in [0.25, 0.5, 1, 2, 3, 5, 6]:
        _lat_count = (coordinates["abs_lat_diff"] <= _threshold).sum()
        _lon_count = (coordinates["abs_lon_diff"] <= _threshold).sum()

        print(
            f"{_threshold}° | "
            f"lat: {_lat_count}/64 ({_lat_count / 64:.1%}) | "
            f"lon: {_lon_count}/64 ({_lon_count / 64:.1%})"
        )
    return


@app.cell
def _(report_findings):
    report_findings(
        """Comparing the absolute magnitudes of the stored coordinates with geocoded city coordinates shows that the geographic values appear to have been perturbed. Only 50.0% of latitude values and 40.6% of longitude values are within 1° of the fetched coordinates, while all locations fall within 6°. The deviations are therefore substantial but bounded, suggesting that the dataset uses altered rather than precise city coordinates.""",
        title="Conclution",
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## **Distance Analysis**
    """)
    return


@app.cell
def _(df, train_test):
    print(train_test.distance.describe())

    print("\nSkewness:", df["distance"].skew())
    return


@app.cell
def _(train_test, within_std_proportions):
    within_std_proportions(train_test.distance, 3)
    return


@app.cell
def _(train_test):
    plt.figure(figsize=(10, 4), dpi=80)
    sns.histplot(data=train_test, x="distance", kde=True)
    plt.title("Distribution of Distance")
    plt.xlabel("Distance (km)")
    plt.show()
    return


@app.cell
def _(df):
    plt.figure(figsize=(10, 1), dpi=80)

    sns.boxplot(data=df, x="distance")
    plt.title("Boxplot of Distance")
    plt.xlabel("Distance (km)")
    plt.show()
    return


@app.cell
def _(col_candidates, cramers_v, find_cluster_driver, train_test):
    _result, _clustered = find_cluster_driver(
        train_test, "distance", col_candidates
    )

    _clustered["month"] = pd.to_datetime(_clustered["date"]).dt.month
    _clustered["day_of_week"] = pd.to_datetime(_clustered["date"]).dt.dayofweek

    _clustered["lane"] = _clustered["pickup"] + " -> " + _clustered["delivery"]
    table = pd.crosstab(_clustered["lane"], _clustered["_cluster"])
    print("lane", cramers_v(table))

    table = pd.crosstab(_clustered["lane"], _clustered["_cluster"])
    print("n unique lanes:", table.shape[0])
    print("n rows:", len(_clustered))

    for _col in ["month", "day_of_week"]:
        _table = pd.crosstab(_clustered[_col], _clustered["_cluster"])
        print(_col, cramers_v(_table))

    _result
    return


@app.cell
def _(report_findings):
    report_findings(
        """Distance's second mode around 2000 km is driven by pickup and delivery location rather than time, equipment, or seasonality, with both showing a moderate association (Cramér's V of 0.36 each) against the distance clusters, while month, day_of_week, and equipment all scored near zero. The raw lane combination scored 0.986, but that's inflated by its 4014 near-unique categories against 48000 rows and isn't a reliable signal on its own."""
    )
    return


@app.cell
def _(report_findings):
    report_findings(
        """Distance's secondary peak around 2000 km is explained by pickup and delivery location, both scoring a moderate Cramér's V of 0.36 against the distance clusters, while equipment, month, and day_of_week all showed negligible association. This indicates distance's multimodality is a geographic effect rather than a temporal or categorical one, and pickup and delivery should be incorporated as features rather than treating distance as a raw standalone input.""",
        title="Conclusion",
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## **Equipment Analysis**
    """)
    return


@app.cell
def _(train_test):
    train_test["equipment"].value_counts(dropna=False)
    return


@app.cell
def _(report_findings):
    report_findings(
        """Equipment showed negligible association with market_index and distance's multimodality (both near zero Cramér's V), and its class distribution is moderately imbalanced (Dry Van 57 percent, Reefer 25 percent, Flatbed 18 percent).""",
        kind="success",
        title="Conclusion",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## **Date Analysis**
    """)
    return


@app.cell
def _(train_test):
    print(
        "date range:", train_test["date"].min(), "to", train_test["date"].max()
    )

    daily_counts = train_test.groupby(train_test["date"].dt.date).size()
    print("days with data:", daily_counts.shape[0])
    return (daily_counts,)


@app.cell
def _(daily_counts):
    plt.figure(figsize=(10, 4))
    daily_counts.plot()
    plt.title("Load Count Over Time")
    plt.ylabel("Count")
    plt.show()
    return


@app.cell
def _(train_test):
    train_test["day_of_week"] = train_test["date"].dt.dayofweek
    plt.figure(figsize=(6, 4))
    sns.countplot(data=train_test, x="day_of_week")
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## **Quote Signal Analysis**
    """)
    return


@app.cell
def _(plot_qq, train_test):
    plot_qq(train_test.quote_signal)
    return


@app.cell
def _(train_test, within_std_proportions):
    within_std_proportions(train_test.quote_signal, 3)
    return


@app.cell
def _(train_test):
    print(train_test.quote_signal.skew())
    return


@app.cell
def _(train_test):
    sns.displot(
        data=train_test,
        x="quote_signal",
        col="equipment",
        col_wrap=3,
        bins=40,
        kde=True,
        facet_kws={"sharey": False},
    )

    plt.show()
    return


@app.cell
def _(report_findings):
    report_findings(
        """Quote_signal shows a leptokurtic distribution, more tightly concentrated around the mean than normal (74.6% within 1sd vs the normal reference of 68%) but with heavier tails, more extreme values beyond 3sd than normal would predict (98.8% vs the 99.7% reference). This is distinct from market_index's near-symmetric mild skew and distance's smooth right-skew, both found earlier, making quote_signal the only one of the three exhibiting fat-tailed, sharp-peaked behavior rather than simple skew.""",
        title="Conclusion",
        kind="success"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## **Posted Rate Analysis**
    """)
    return


@app.cell
def _(plot_qq, train_test):
    plot_qq(train_test.posted_rate)
    return


@app.cell
def _(train_test, within_std_proportions):
    within_std_proportions(train_test.posted_rate, 3)
    return


@app.cell
def _(train_test):
    print(train_test.posted_rate.skew())
    return


@app.cell
def _(train_test):
    sns.displot(
        data=train_test,
        x="posted_rate",
        col="equipment",
        col_wrap=3,
        bins=40,
        kde=True,
        facet_kws={"sharey": False},
    )

    plt.show()
    return


@app.cell
def _(train_test):
    plt.figure(figsize=(4, 4))

    sns.boxplot(data=train_test, x="equipment", y="posted_rate")

    plt.show()
    return


@app.cell
def _(train_test):
    train_test["month"] = train_test["date"].dt.month
    train_test["day_of_week"] = train_test["date"].dt.dayofweek

    train_test.nlargest(15, "posted_rate")[
        [
            "load_id",
            "pickup",
            "delivery",
            "distance",
            "weight",
            "equipment",
            "posted_rate",
        ]
    ]
    return


@app.cell
def _(train_test):
    train_test["spread"] = (
        train_test["quote_signal"] - train_test["market_index"]
    )
    train_test[
        [
            "distance",
            "weight",
            "market_index",
            "quote_signal",
            "spread",
            "posted_rate",
        ]
    ].corr()["posted_rate"]
    return


@app.cell
def _(train_test):
    daily_rate = train_test.groupby(train_test["date"].dt.date)[
        "posted_rate"
    ].mean()
    plt.figure(figsize=(10, 4))
    daily_rate.plot()
    plt.title("Mean Posted Rate Over Time")
    plt.show()
    return


@app.cell
def _(train_test):
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=train_test, x="month", y="posted_rate", showfliers=False)
    plt.title("Posted Rate by Month (outliers hidden)")
    plt.show()
    return


@app.cell
def _(train_test):
    plt.figure(figsize=(6, 4))
    sns.boxplot(
        data=train_test, x="day_of_week", y="posted_rate", showfliers=False
    )
    plt.title("Posted Rate by Day of Week (outliers hidden)")
    plt.show()
    return


@app.cell
def _(train_test):
    train_test.groupby("equipment")["posted_rate"].describe()
    return


@app.cell
def _(train_test):
    plt.figure(figsize=(5, 4))
    sns.scatterplot(
        data=train_test, x="distance", y="posted_rate", alpha=0.2, s=8
    )
    plt.show()
    return


@app.cell
def _(train_test):
    plt.figure(figsize=(5, 4))
    sns.scatterplot(
        data=train_test, x="weight", y="posted_rate", alpha=0.2, s=8
    )
    plt.show()
    return


@app.cell
def _(train_test):
    train_test["rate_per_km"] = (
        train_test["posted_rate"] / train_test["distance"]
    )
    plt.figure(figsize=(4, 4), dpi=80)
    sns.histplot(train_test["rate_per_km"], bins=40, kde=True)
    plt.show()
    return


@app.cell
def _(train_test):
    print(train_test["rate_per_km"].describe())
    print("skew:", train_test["rate_per_km"].skew())
    return


@app.cell
def _(train_test):
    print("unique pickups:", train_test["pickup"].nunique())
    print("unique deliveries:", train_test["delivery"].nunique())
    print(train_test["pickup"].value_counts().head(10))
    print(train_test["delivery"].value_counts().head(10))
    return


@app.cell
def _(find_cluster_driver, train_test):
    mi_results, mi_clustered = find_cluster_driver(
        train_test,
        "posted_rate",
        ["equipment", "pickup", "delivery", "month", "day_of_week"],
        n_components=2,
    )
    print(mi_results)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## **Conclusion**


    The exploratory analysis across the dataset points to a coherent story once all the individual findings are put together. Weight is close to normally distributed in its center, with about 67.18% of values falling within one standard deviation and 94.93% within two, closely tracking the theoretical 68% and 95% one would expect from a true normal distribution, and the Q-Q plot backs this up through the middle of the range. The deviation shows up only at the tails, which makes sense given weight has hard physical lower and upper limits and a visible pile-up near the maximum around 47,500 lbs. Neither distance nor equipment type explains why weight varies the way it does, weights are scattered broadly at almost every distance value, and Dry Van, Reefer, and Flatbed all carry nearly identical weight profiles, a mean and median near 31,000 lbs and a standard deviation around 9,000 in every case. That similarity across equipment types suggests weight is being driven by something not captured in these two columns, possibly cargo type, shipper behavior, or some other unmeasured characteristic of the load itself. The small number of negative and missing weight values, only about 0.6% of the data, were dropped rather than filled in, since weight appears to shift with time and imputing a value would mean assuming conditions that weren't actually observed on that date.

    Market_index tells a related but distinct story. Rather than one smooth distribution, it has three separate peaks around 1.0, 1.2, and 1.35, a shape that a simple skew value of 0.214 completely fails to capture since skew only measures asymmetry, not the number of modes. Testing this against equipment type showed the three peaks appear identically in Dry Van, Reefer, and Flatbed alike, with only the row counts scaling to match each category's size, which rules out equipment as the explanation. Testing instead against time, monthly, biweekly, and triweekly windows, showed the opposite, each narrower time window collapses into something close to a single mode on its own, and the specific level that mode sits at drifts across the year, starting lower around January, climbing through the middle of the year, then coming back down by September and October. In other words, the three peaks in the pooled data are really just the same underlying seasonal curve sampled at different points and stacked on top of each other, not three separate fixed categories coexisting at once. Of the 374 missing market_index values, only 24 could be recovered by matching another row with the same pickup, delivery, and date, a recovery rate of roughly 6.4%, but those matches were judged the most trustworthy source available since they reflect the same lane under the same day's market conditions, so the median of the matches was used to fill those specific rows.

    Distance carries its own separate finding. It is moderately right skewed at 0.77, with a long tail of longer hauls pulling the mean above the median, and the histogram reveals a genuine second cluster of loads around 2000 to 2100 kilometers sitting on top of the broader skewed curve. Testing this bump against the same set of candidate columns showed pickup and delivery location are what explain it, both scoring a real association around 0.36, while equipment, month, and day of week all came back essentially unrelated. So the extra bump in the distance distribution reflects certain regions consistently producing longer hauls, not a seasonal or equipment driven effect.

    Quote_signal, the third numeric market feature, has yet another distinct shape. It's more tightly bunched around its mean than a normal distribution would predict, about 74.6% falling within one standard deviation against a 68% reference, but its tails are fatter than normal too, with more extreme values beyond three standard deviations than expected, giving it the classic leptokurtic profile that showed up clearly as an S-curve in the Q-Q plot. This pattern held consistently across all three equipment types.

    Turning to posted_rate, the actual target, it is by far the most heavily skewed of anything examined so far at 1.90, with a sharp concentration of loads in the low range and a long thin tail stretching out toward 25,000. Distance turns out to be its dominant driver, correlating at 0.91, while weight, market_index, quote_signal, and even the difference between quote_signal and market_index all correlate with posted_rate at levels close to zero, none exceeding roughly 0.05 in magnitude. Since distance itself is shaped by pickup and delivery location, it isn't surprising that location also comes out as the strongest categorical driver of posted_rate directly, at almost the same strength seen for distance. Equipment plays a smaller but genuine role here, unlike its negligible effect on market_index and distance, Reefer loads average about 280 higher in rate than Dry Van, a gap too consistent to dismiss even though it's modest next to location's effect. Time factors, month and day of week, show essentially no relationship with posted_rate, mirroring what was found with market_index's seasonality not carrying through to the target. Looking closely at the distance versus posted_rate scatter plot also revealed the relationship isn't one clean line, there's a dense central band where most loads sit, but also a thinner band running below it and a sparser scatter running well above it for the same distance values, worth a direct look at those specific rows before modeling to judge whether they represent genuine alternate pricing arrangements or possible data issues.

    Taken together, the picture that emerges is one where location, through pickup and delivery, is the thread running underneath almost everything, it explains distance's bimodal shape and it explains most of posted_rate's variation as well. Time explains market_index's structure but conspicuously does not explain the target itself. Equipment has little bearing on any of the other features but does carry a real, if modest, direct effect on price. And weight and quote_signal, despite having their own well defined and interesting distributional shapes, don't show a meaningful linear relationship with rate on their own, meaning if they contribute to prediction at all it will likely be through nonlinear or interaction effects rather than as simple standalone predictors.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
