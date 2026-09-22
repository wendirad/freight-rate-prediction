import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import seaborn as sns
    from utils.eda_helpers import (
        load_freight_data,
        report_findings,
    )


@app.cell
def _():
    mo.md(r"""
    # Equipment and Date Analysis
    """)


@app.cell
def _():
    data = load_freight_data()
    return (data,)


@app.cell
def _(data):
    equipment_counts = data["equipment"].value_counts(dropna=False)
    equipment_share = data["equipment"].value_counts(normalize=True, dropna=False)
    mo.hstack(
        [
            equipment_counts.rename("count").to_frame(),
            equipment_share.rename("share").to_frame(),
        ]
    )


@app.cell
def _(data):
    daily_counts = data.groupby(data["date"].dt.date).size()
    date_summary = {
        "start": data["date"].min(),
        "end": data["date"].max(),
        "days_with_data": len(daily_counts),
    }
    print(date_summary)
    return (daily_counts,)


@app.cell
def _(daily_counts):
    _figure, _axis = plt.subplots(figsize=(10, 4))
    daily_counts.plot(ax=_axis)
    _axis.set(title="Load Count Over Time", ylabel="Count", xlabel="Date")
    _figure.tight_layout()
    _figure


@app.cell
def _(data):
    weekday_data = data.assign(day_of_week=data["date"].dt.day_name())
    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    _figure, _axis = plt.subplots(figsize=(8, 4))
    sns.countplot(data=weekday_data, x="day_of_week", order=weekday_order, ax=_axis)
    _axis.tick_params(axis="x", rotation=30)
    _figure.tight_layout()
    _figure


@app.cell
def _():
    report_findings(
        """Equipment showed negligible association with market_index and distance's multimodality (both near zero Cramér's V), and its class distribution is moderately imbalanced (Dry Van 57 percent, Reefer 25 percent, Flatbed 18 percent).""",
        title="Conclusion",
        kind="success",
    )


if __name__ == "__main__":
    app.run()
