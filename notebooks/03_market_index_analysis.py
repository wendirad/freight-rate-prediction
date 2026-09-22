import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import pandas as pd
    import seaborn as sns
    from freight_rate_prediction_challenge.eda_helpers import (
        EDAConfig,
        cramers_v,
        find_cluster_driver,
        load_freight_data,
        plot_qq,
        recover_market_index,
        report_findings,
        within_std_proportions,
    )


@app.cell
def _():
    mo.md(r"""
    # Market Index Analysis
    """)


@app.cell
def _():
    cfg = EDAConfig()
    data = load_freight_data(cfg)
    market_index = data["market_index"].dropna()
    return cfg, data, market_index


@app.cell
def _(market_index):
    mo.hstack([plot_qq(market_index), within_std_proportions(market_index)])


@app.cell
def _(data):
    recovered_data, recovered_values = recover_market_index(data)
    recovery_summary = {
        "missing_before": int(data["market_index"].isna().sum()),
        "recovered": int(recovered_values.notna().sum()),
        "remaining_missing": int(recovered_data["market_index"].isna().sum()),
    }
    recovery_summary
    return (recovered_data,)


@app.cell
def _():
    report_findings(
        """Among the 374 observations with missing market_index, only 24 could be matched with another observation having the same pickup, delivery, and date, giving a recovery rate of approximately 6.42%. Although this method provides limited coverage, these matches are the most suitable for recovery because they represent the same transportation lane under the same day's market conditions. The median market_index of matching observations was therefore used to recover these values."""
    )


@app.cell
def _(cfg):
    category = mo.ui.dropdown(
        options=list(cfg.categorical_cols),
        value="equipment",
        label="Facet by",
    )
    category
    return (category,)


@app.cell
def _(category, data):
    chart = sns.displot(
        data=data,
        x="market_index",
        col=category.value,
        col_wrap=3,
        bins=40,
        kde=True,
        facet_kws={"sharey": False},
        height=3,
    )
    chart.figure


@app.cell
def _():
    report_findings(
        """The market index distribution is multimodal, with three visible peaks around 1.0, 1.2, and 1.35, rather than a single unimodal shape, despite an overall skew of only 0.214 which masks this structure. The main mass sits between roughly 0.9 and 1.2, with a longer tail extending toward 1.45 to 1.5, and the boxplot shows no flagged outliers with whiskers spanning the full observed range.

The market index's multimodal pattern persists identically across all three equipment types (Dry Van, Reefer, Flatbed), with only the counts scaling to match each category's sample size, ruling out equipment type as the source of the multimodality."""
    )


@app.cell
def _(cfg, data):
    candidates = (*cfg.categorical_cols, *cfg.date_cols)
    association, clustered_market = find_cluster_driver(
        data, "market_index", candidates
    )
    enriched_clusters = clustered_market.assign(
        month=clustered_market["date"].dt.month,
        day_of_week=clustered_market["date"].dt.dayofweek,
        lane=clustered_market["pickup"] + " -> " + clustered_market["delivery"],
    )
    extra_associations = pd.DataFrame(
        {
            "column": ["month", "day_of_week", "lane"],
            "cramers_v": [
                cramers_v(
                    pd.crosstab(enriched_clusters[col], enriched_clusters["_cluster"])
                )
                for col in ["month", "day_of_week", "lane"]
            ],
            "n_categories": [
                enriched_clusters[col].nunique()
                for col in ["month", "day_of_week", "lane"]
            ],
        }
    )
    pd.concat([association, extra_associations]).sort_values(
        "cramers_v", ascending=False
    )
    return (enriched_clusters,)


@app.cell
def _():
    time_granularity = mo.ui.dropdown(
        options=["month", "biweek", "triweek"],
        value="month",
        label="Time window",
    )
    time_granularity
    return (time_granularity,)


@app.cell
def _(data, time_granularity):
    start = data["date"].min()
    days_elapsed = (data["date"] - start).dt.days
    temporal_data = data.assign(
        month=data["date"].dt.month,
        biweek=days_elapsed // 14,
        triweek=days_elapsed // 21,
    )
    temporal_chart = sns.displot(
        data=temporal_data,
        x="market_index",
        col=time_granularity.value,
        col_wrap=5,
        bins=40,
        kde=True,
        facet_kws={"sharey": False},
    )
    temporal_chart.figure


@app.cell
def _():
    report_findings(
        """The market_index's multimodal pattern holds up at every time resolution tested, monthly, biweekly, and triweekly, with the same peaks recurring inside each window rather than resolving into a single mode. This rules out calendar time granularity as the source of the multimodality."""
    )


@app.cell
def _(data):
    missing_rate = (
        data.assign(missing=data["market_index"].isna())
        .groupby("equipment")["missing"]
        .mean()
        .rename("missing_rate")
    )
    missing_rate.to_frame()


@app.cell
def _():
    report_findings(
        """
After recovering the reliable same-lane and same-date observations, the remaining missing values represent only a small proportion of the dataset. Missing rates are approximately 0.74% for Dry Van, 0.83% for Flatbed, and 0.82% for Reefer, showing similarly low missingness across equipment types. Since estimating the remaining values would require increasingly uncertain temporal assumptions, dropping these observations is expected to have substantially less impact on the dataset than introducing potentially inaccurate imputed values.""",
        title="Conclusion",
        kind="success",
    )


if __name__ == "__main__":
    app.run()
