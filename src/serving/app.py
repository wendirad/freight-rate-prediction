"""Streamlit interface for single and batch freight-rate predictions."""

from __future__ import annotations

import os
import pickle
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from serving.inference import PredictionBundle

DEFAULT_BUNDLE_PATH = "artifacts/model.joblib"


@st.cache_resource
def load_bundle(path: str, modified_at: int) -> PredictionBundle:
    del modified_at
    return PredictionBundle.load(path)


def single_prediction_form(bundle: PredictionBundle) -> None:
    equipment_options = bundle.categorical_options.get(
        "equipment", ["Dry Van", "Flatbed", "Reefer"]
    )

    with st.form("single-prediction"):
        st.subheader("Shipment")
        left, right = st.columns(2)
        with left:
            load_id = st.text_input("Load ID", value="web-single")
            pickup = st.text_input("Pickup")
            pickup_lat = st.number_input("Pickup latitude", format="%.6f")
            pickup_lon = st.number_input("Pickup longitude", format="%.6f")
            distance = st.number_input("Distance", min_value=0.0)
            equipment = st.selectbox("Equipment", equipment_options)
        with right:
            delivery = st.text_input("Delivery")
            delivery_lat = st.number_input("Delivery latitude", format="%.6f")
            delivery_lon = st.number_input("Delivery longitude", format="%.6f")
            weight = st.number_input("Weight", min_value=0.0)
            shipment_date = st.date_input("Date", value=datetime.now(UTC).date())
            market_index = st.number_input("Market index", format="%.6f")
            quote_signal = st.number_input("Quote signal", format="%.6f")

        submitted = st.form_submit_button("Predict rate", type="primary")

    if not submitted:
        return

    row = pd.DataFrame(
        [
            {
                "load_id": load_id,
                "pickup": pickup,
                "delivery": delivery,
                "pickup_lat": pickup_lat,
                "pickup_lon": pickup_lon,
                "delivery_lat": delivery_lat,
                "delivery_lon": delivery_lon,
                "distance": distance,
                "equipment": equipment,
                "weight": weight,
                "date": shipment_date,
                "market_index": market_index,
                "quote_signal": quote_signal,
            }
        ]
    )
    try:
        result = bundle.predict(row)
    except (KeyError, TypeError, ValueError) as exc:
        st.error(f"Could not produce a prediction: {exc}")
        return

    prediction = float(result.loc[result.index[0], bundle.prediction_col])
    st.metric("Predicted posted rate", f"${prediction:,.2f}")


def batch_prediction_form(bundle: PredictionBundle) -> None:
    st.write("Upload a CSV containing these columns:")
    st.code(", ".join(bundle.input_columns), language=None)
    uploaded = st.file_uploader("Freight CSV", type="csv")
    if uploaded is None:
        return

    try:
        rows = pd.read_csv(uploaded)
        result = bundle.predict(rows)
    except (KeyError, TypeError, ValueError) as exc:
        st.error(f"Could not process the CSV: {exc}")
        return

    st.success(f"Predicted {len(result):,} rows.")
    st.dataframe(result.head(100), use_container_width=True)
    st.download_button(
        "Download predictions",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name="freight-rate-predictions.csv",
        mime="text/csv",
        type="primary",
    )


def main() -> None:
    st.set_page_config(page_title="Freight Rate Predictor", page_icon="🚚")
    st.title("Freight Rate Predictor")
    st.caption("Estimate one shipment or score a CSV with the same fitted pipeline.")

    bundle_path = Path(os.environ.get("MODEL_BUNDLE_PATH", DEFAULT_BUNDLE_PATH))
    if not bundle_path.is_file():
        st.error(f"Model bundle not found at {bundle_path}.")
        st.code("make artifact", language="bash")
        st.stop()

    try:
        bundle = load_bundle(str(bundle_path), bundle_path.stat().st_mtime_ns)
    except (
        AttributeError,
        EOFError,
        ImportError,
        OSError,
        TypeError,
        ValueError,
        pickle.UnpicklingError,
    ) as exc:
        st.error(f"Could not load the model bundle: {exc}")
        st.stop()

    single_tab, batch_tab = st.tabs(["Single prediction", "Batch CSV"])
    with single_tab:
        single_prediction_form(bundle)
    with batch_tab:
        batch_prediction_form(bundle)


main()
