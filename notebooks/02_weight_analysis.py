import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")

with app.setup:
    import altair as alt
    import marimo as mo
    import matplotlib.pyplot as plt
    import seaborn as sns

    from freight_rate_prediction_challenge.eda_helpers import (
        load_freight_data,
        plot_qq,
        report_findings,
        within_std_proportions,
    )

    alt.data_transformers.enable("vegafusion")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Weight Analysis
    """)
    return


@app.cell
def _():
    data = load_freight_data()
    valid_weight = data.loc[data["weight"] > 0, "weight"].dropna()
    return data, valid_weight


@app.cell
def _(valid_weight):
    plot_qq(valid_weight)
    return


@app.cell
def _(valid_weight):
    within_std_proportions(valid_weight)
    return


@app.cell(hide_code=True)
def _():
    report_findings(
        """After ignoring the invalid negative values, 67.18% of the weight observations fall within one standard deviation of the mean and 94.93% fall within two standard deviations. These are very close to the theoretical 68% and 95% expected for a normal distribution. The Q-Q plot supports this in the central portion of the data, where observations closely follow the reference line. However, deviations at both ends indicate that the distribution is not perfectly normal, mainly because weight has clear lower and upper boundaries, including a concentration near the maximum weight of 47,500."""
    )
    return


@app.cell
def _(data):
    sample = data.sample(n=min(5_000, len(data)), random_state=0)
    scatter = (
        alt.Chart(sample)
        .mark_circle(opacity=0.45)
        .encode(
            x="distance:Q",
            y="weight:Q",
            color="equipment:N",
            tooltip=["distance", "weight", "equipment"],
        )
        .properties(width=700, height=320, title="Distance vs Weight")
    )
    scatter
    return


@app.cell(hide_code=True)
def _():
    report_findings(
        """The combined analysis of distance and equipment type shows that neither variable strongly explains the variation in weight. Across Dry Van, Reefer, and Flatbed, weights remain widely distributed at nearly every distance, with substantial overlap between equipment types. There is no clear trend of weight increasing or decreasing with distance, and separating the data by equipment produces very similar patterns. This suggests that weight is likely driven by other shipment characteristics or interactions not captured by these two features alone."""
    )
    return


@app.cell
def _(data):
    distribution = sns.displot(
        data=data,
        x="weight",
        col="equipment",
        col_wrap=3,
        bins=40,
        kde=True,
        facet_kws={"sharey": False},
    )
    distribution.figure
    return


@app.cell
def _(data):
    figure, axis = plt.subplots(figsize=(5, 4))
    sns.boxplot(data=data, x="equipment", y="weight", ax=axis)
    figure.tight_layout()
    figure
    return


@app.cell
def _(data):
    data.groupby("equipment")["weight"].describe(
        percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]
    )
    return


@app.cell(hide_code=True)
def _():
    report_findings(
        """The weight distribution is nearly identical across Dry Van, Reefer, and Flatbed equipment. All three have a mean and median around 31K and a standard deviation of roughly 9K, with very similar ranges and percentiles. This suggests that equipment type alone does not have a meaningful effect on weight, and the variation in weight is likely explained by other features or combinations of features in the dataset."""
    )
    return


@app.cell
def _(data, valid_weight):
    weight_quality = {
        "negative_or_zero": int((data["weight"] <= 0).sum()),
        "missing": int(data["weight"].isna().sum()),
        "skew_valid_weight": float(valid_weight.skew()),
    }
    missing_by_equipment = (
        data.assign(weight_missing=data["weight"].isna())
        .groupby("equipment")["weight_missing"]
        .mean()
        .rename("missing_rate")
    )
    mo.vstack([weight_quality, missing_by_equipment.to_frame()])
    return


@app.cell(hide_code=True)
def _():
    report_findings(
        """Removing negative weight observations had no meaningful effect on the weight distribution, with the proportions within one and two standard deviations remaining approximately 67.18% and 94.93%, respectively. Since missing weights represent only about 0.6% of the data and weight is also temporally dependent, attempting to recover or impute these values could introduce inaccurate assumptions by ignoring conditions specific to the date of each observation. Therefore, dropping these records was considered more appropriate than imputation.""",
        title="Conclusion",
        kind="success",
    )
    return


if __name__ == "__main__":
    app.run()
