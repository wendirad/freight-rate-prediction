import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import seaborn as sns
    from utils.eda_helpers import (
        load_freight_data,
        plot_qq,
        report_findings,
        within_std_proportions,
    )


@app.cell
def _():
    mo.md(r"""
    # Quote Signal Analysis
    """)


@app.cell
def _():
    data = load_freight_data()
    quote_signal = data["quote_signal"]
    return data, quote_signal


@app.cell
def _(quote_signal):
    mo.hstack([plot_qq(quote_signal), within_std_proportions(quote_signal)])


@app.cell
def _(quote_signal):
    {
        "skew": float(quote_signal.skew()),
        "excess_kurtosis": float(quote_signal.kurtosis()),
    }


@app.cell
def _(data):
    chart = sns.displot(
        data=data,
        x="quote_signal",
        col="equipment",
        col_wrap=3,
        bins=40,
        kde=True,
        facet_kws={"sharey": False},
    )
    chart.figure


@app.cell
def _():
    report_findings(
        """Quote_signal shows a leptokurtic distribution, more tightly concentrated around the mean than normal (74.6% within 1sd vs the normal reference of 68%) but with heavier tails, more extreme values beyond 3sd than normal would predict (98.8% vs the 99.7% reference). This is distinct from market_index's near-symmetric mild skew and distance's smooth right-skew, both found earlier, making quote_signal the only one of the three exhibiting fat-tailed, sharp-peaked behavior rather than simple skew.""",
        title="Conclusion",
        kind="success",
    )


if __name__ == "__main__":
    app.run()
