import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import seaborn as sns

    from freight_rate_prediction_challenge.eda_helpers import (
        find_cluster_driver,
        load_freight_data,
        plot_qq,
        within_std_proportions,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Posted Rate Analysis
    """)


@app.cell
def _():
    data = load_freight_data()
    target_data = data.assign(
        month=data["date"].dt.month,
        day_of_week=data["date"].dt.dayofweek,
        spread=data["quote_signal"] - data["market_index"],
        rate_per_km=data["posted_rate"] / data["distance"],
    )
    return (target_data,)


@app.cell
def _(target_data):
    mo.hstack(
        [
            plot_qq(target_data["posted_rate"]),
            within_std_proportions(target_data["posted_rate"]),
        ]
    )


@app.cell
def _(target_data):
    distribution = sns.displot(
        data=target_data,
        x="posted_rate",
        col="equipment",
        col_wrap=3,
        bins=40,
        kde=True,
        facet_kws={"sharey": False},
    )
    distribution.figure


@app.cell
def _(target_data):
    _figure, _axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.scatterplot(
        data=target_data, x="distance", y="posted_rate", alpha=0.2, s=8, ax=_axes[0]
    )
    sns.scatterplot(
        data=target_data, x="weight", y="posted_rate", alpha=0.2, s=8, ax=_axes[1]
    )
    _figure.tight_layout()
    _figure


@app.cell
def _(target_data):
    correlations = (
        target_data[
            [
                "distance",
                "weight",
                "market_index",
                "quote_signal",
                "spread",
                "posted_rate",
            ]
        ]
        .corr()["posted_rate"]
        .sort_values(ascending=False)
    )
    correlations.to_frame("correlation_with_posted_rate")


@app.cell
def _(target_data):
    _figure, _axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.boxplot(
        data=target_data, x="month", y="posted_rate", showfliers=False, ax=_axes[0]
    )
    sns.boxplot(
        data=target_data,
        x="day_of_week",
        y="posted_rate",
        showfliers=False,
        ax=_axes[1],
    )
    _axes[0].set_title("Posted Rate by Month")
    _axes[1].set_title("Posted Rate by Day of Week")
    _figure.tight_layout()
    _figure


@app.cell
def _(target_data):
    mo.hstack(
        [
            target_data.groupby("equipment")["posted_rate"].describe(),
            target_data["rate_per_km"].describe().to_frame(),
        ]
    )


@app.cell
def _(target_data):
    driver_results, _ = find_cluster_driver(
        target_data,
        "posted_rate",
        ["equipment", "pickup", "delivery", "month", "day_of_week"],
        n_components=2,
    )
    driver_results


@app.cell
def _(target_data):
    daily_rate = target_data.groupby(target_data["date"].dt.date)["posted_rate"].mean()
    _figure, _axis = plt.subplots(figsize=(10, 4))
    daily_rate.plot(ax=_axis)
    _axis.set(
        title="Mean Posted Rate Over Time", ylabel="Mean posted rate", xlabel="Date"
    )
    _figure.tight_layout()
    _figure


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


if __name__ == "__main__":
    app.run()
