import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from geopy.extra.rate_limiter import RateLimiter
    from geopy.geocoders import Nominatim
    from scipy.stats import linregress
    from utils.eda_helpers import (
        EDAConfig,
        cramers_v,
        find_cluster_driver,
        haversine,
        load_freight_data,
        report_findings,
        within_std_proportions,
    )


@app.cell
def _():
    mo.md(r"""
    # Geography and Distance Analysis

    The external geocoding comparison is opt-in so opening this notebook does not
    automatically make network requests or wait on rate limits.
    """)
    return


@app.cell
def _():
    cfg = EDAConfig()
    data = load_freight_data(cfg)
    location_rows = pd.concat(
        [
            data[["pickup", "pickup_lat", "pickup_lon"]].rename(
                columns={"pickup": "location", "pickup_lat": "lat", "pickup_lon": "lon"}
            ),
            data[["delivery", "delivery_lat", "delivery_lon"]].rename(
                columns={
                    "delivery": "location",
                    "delivery_lat": "lat",
                    "delivery_lon": "lon",
                }
            ),
        ],
        ignore_index=True,
    )
    return cfg, data, location_rows


@app.cell
def _(location_rows):
    coordinate_consistency = location_rows.groupby("location").agg(
        observations=("location", "size"),
        unique_lat=("lat", "nunique"),
        unique_lon=("lon", "nunique"),
    )
    coordinate_summary = {
        "unique_locations": int(location_rows["location"].nunique()),
        "unique_coordinate_pairs": int(
            location_rows[["lat", "lon"]].drop_duplicates().shape[0]
        ),
        "locations_with_multiple_coordinates": int(
            (
                (coordinate_consistency["unique_lat"] > 1)
                | (coordinate_consistency["unique_lon"] > 1)
            ).sum()
        ),
    }
    coordinate_summary
    return


@app.cell
def _():
    report_findings(
        """The dataset contains 64 unique locations and 64 unique latitude/longitude pairs, with no missing geographic information. Each pickup or delivery location consistently maps to exactly one coordinate pair, with no locations associated with multiple coordinates. This indicates that the geographic variables are internally consistent and that the latitude/longitude columns provide a reliable numerical representation of the pickup and delivery locations."""
    )
    return


@app.cell
def _(data):
    distance_data = data.assign(
        geo_distance=haversine(
            data["pickup_lat"],
            data["pickup_lon"],
            data["delivery_lat"],
            data["delivery_lon"],
        )
    )
    distance_data = distance_data.assign(
        distance_diff=distance_data["distance"] - distance_data["geo_distance"],
        distance_ratio=distance_data["distance"] / distance_data["geo_distance"],
    )
    distance_data[
        ["distance", "geo_distance", "distance_diff", "distance_ratio"]
    ].describe()
    return (distance_data,)


@app.cell
def _():
    report_findings(
        """When the straight-line geographic distance between pickup and delivery was calculated from the coordinates, it showed an extremely strong correlation of 0.9995 with the provided distance variable. This confirms that the coordinates and recorded distances represent highly consistent geographic relationships. However, the calculated geographic distance was, on average, approximately 419 km greater than the provided distance."""
    )
    return


@app.cell
def _(distance_data):
    regression = linregress(distance_data["geo_distance"], distance_data["distance"])
    {
        "correlation": distance_data["distance"].corr(distance_data["geo_distance"]),
        "slope": regression.slope,
        "intercept": regression.intercept,
        "r_squared": regression.rvalue**2,
    }
    return


@app.cell
def _():
    report_findings(
        """The provided distance has an almost perfect linear relationship with the coordinate-derived geographic distance (R² = 0.9991). The provided distances are typically around 73–74% of the calculated geographic distances, suggesting a systematic difference in scaling or calculation method or a random geographic inconsistencies."""
    )
    return


@app.cell
def _(data):
    mo.hstack(
        [
            within_std_proportions(data["distance"]),
            data["distance"].describe().to_frame(),
        ]
    )
    return


@app.cell
def _(data):
    figure, axes = plt.subplots(2, 1, figsize=(10, 5), height_ratios=[4, 1])
    sns.histplot(data=data, x="distance", kde=True, ax=axes[0])
    sns.boxplot(data=data, x="distance", ax=axes[1])
    figure.tight_layout()
    figure
    return


@app.cell
def _(cfg, data):
    candidates = (*cfg.categorical_cols, *cfg.date_cols)
    distance_association, distance_clusters = find_cluster_driver(
        data, "distance", candidates
    )
    enriched_distance_clusters = distance_clusters.assign(
        month=distance_clusters["date"].dt.month,
        day_of_week=distance_clusters["date"].dt.dayofweek,
        lane=distance_clusters["pickup"] + " -> " + distance_clusters["delivery"],
    )
    extras = pd.DataFrame(
        {
            "column": ["month", "day_of_week", "lane"],
            "cramers_v": [
                cramers_v(
                    pd.crosstab(
                        enriched_distance_clusters[col],
                        enriched_distance_clusters["_cluster"],
                    )
                )
                for col in ["month", "day_of_week", "lane"]
            ],
            "n_categories": [
                enriched_distance_clusters[col].nunique()
                for col in ["month", "day_of_week", "lane"]
            ],
        }
    )
    pd.concat([distance_association, extras]).sort_values("cramers_v", ascending=False)
    return


@app.cell
def _():
    report_findings(
        """Distance's second mode around 2000 km is driven by pickup and delivery location rather than time, equipment, or seasonality, with both showing a moderate association (Cramér's V of 0.36 each) against the distance clusters, while month, day_of_week, and equipment all scored near zero. The raw lane combination scored 0.986, but that's inflated by its 4014 near-unique categories against 48000 rows and isn't a reliable signal on its own."""
    )
    return


@app.cell
def _():
    run_geocoding = mo.ui.run_button(label="Compare with live Nominatim coordinates")
    run_geocoding
    return (run_geocoding,)


@app.cell
def _(location_rows, run_geocoding):
    mo.stop(
        not run_geocoding.value,
        mo.md("Live coordinate comparison has not been requested."),
    )
    geolocator = Nominatim(user_agent="freight_rate_location_analysis")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    fetched = []
    for city in sorted(location_rows["location"].unique()):
        match = geocode(f"{city}, USA", exactly_one=True, country_codes="us")
        fetched.append(
            {
                "location": city,
                "fetched_lat": match.latitude if match else None,
                "fetched_lon": match.longitude if match else None,
            }
        )
    live_coordinates = pd.DataFrame(fetched)
    stored_coordinates = location_rows.drop_duplicates("location")
    coordinate_comparison = stored_coordinates.merge(live_coordinates, on="location")
    coordinate_comparison["abs_lat_diff"] = (
        coordinate_comparison["lat"].abs() - coordinate_comparison["fetched_lat"].abs()
    ).abs()
    coordinate_comparison["abs_lon_diff"] = (
        coordinate_comparison["lon"].abs() - coordinate_comparison["fetched_lon"].abs()
    ).abs()
    coordinate_comparison
    return


@app.cell
def _():
    report_findings(
        """Comparing the absolute magnitudes of the stored coordinates with geocoded city coordinates shows that the geographic values appear to have been perturbed. Only 50.0% of latitude values and 40.6% of longitude values are within 1° of the fetched coordinates, while all locations fall within 6°. The deviations are therefore substantial but bounded, suggesting that the dataset uses altered rather than precise city coordinates.""",
        title="Conclusion",
        kind="success",
    )
    return


@app.cell
def _():
    report_findings(
        """Distance's secondary peak around 2000 km is explained by pickup and delivery location, both scoring a moderate Cramér's V of 0.36 against the distance clusters, while equipment, month, and day_of_week all showed negligible association. This indicates distance's multimodality is a geographic effect rather than a temporal or categorical one, and pickup and delivery should be incorporated as features rather than treating distance as a raw standalone input.""",
        title="Conclusion",
        kind="success",
    )
    return


if __name__ == "__main__":
    app.run()
