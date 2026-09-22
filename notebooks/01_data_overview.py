import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    from freight_rate_prediction_challenge.eda_helpers import (
        EDAConfig,
        load_freight_data,
        report_findings,
    )


@app.cell
def _():
    mo.md(r"""
    # Freight Rate Prediction: Data Overview

    This notebook validates the dataset schema and reviews its basic quality.
    The feature-specific investigations live in the numbered notebooks that follow.

    | Column | Description |
    | --- | --- |
    | `load_id` | Identifier for a shipment load |
    | `pickup`, `delivery` | Origin and destination names |
    | `pickup_lat`, `pickup_lon` | Pickup coordinates |
    | `delivery_lat`, `delivery_lon` | Delivery coordinates |
    | `distance` | Shipment distance in kilometres |
    | `equipment` | Carrier equipment type |
    | `weight` | Load weight in pounds |
    | `date` | Shipment date |
    | `market_index` | Market supply-and-demand cost index |
    | `quote_signal` | Live truck/cargo balance indicator |
    | `posted_rate` | Prediction target: total shipment rate |
    """)
    return


@app.cell
def _():
    cfg = EDAConfig()
    data = load_freight_data(cfg)
    print(f"Shape: {data.shape}")
    return (data,)


@app.cell
def _(data):
    with mo.redirect_stdout():
        data.info()
    return


@app.cell
def _(data):
    quality_summary = {
        "exact_duplicate_rows": int(data.duplicated().sum()),
        "duplicate_load_ids": int(data["load_id"].duplicated().sum()),
        "missing_cells": int(data.isna().sum().sum()),
    }
    quality_summary
    return


@app.cell
def _(data):
    missing = data.isna().sum().rename("missing_count")
    missing[missing > 0].to_frame()
    return


@app.cell
def _(data):
    data.describe(include="all")
    return


@app.cell
def _():
    report_findings(
        "The dataset has 48,000 rows and 14 expected columns. Missing values are "
        "limited to 'weight' and 'market_index'; the dedicated notebooks examine "
        "those cases before deciding how to handle them.",
        title="Next steps",
    )
    return


if __name__ == "__main__":
    app.run()
