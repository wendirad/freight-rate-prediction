"""Generate the two files required by the assessment scorer."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from serving.inference import PredictionBundle


def _city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    locations = []
    for frame in frames:
        locations.extend(
            [
                frame[["pickup", "pickup_lat", "pickup_lon"]].rename(
                    columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}
                ),
                frame[["delivery", "delivery_lat", "delivery_lon"]].rename(
                    columns={
                        "delivery": "city",
                        "delivery_lat": "lat",
                        "delivery_lon": "lon",
                    }
                ),
            ]
        )
    return pd.concat(locations, ignore_index=True).groupby("city")[["lat", "lon"]].median()


def generate(
    bundle_path: Path,
    train_path: Path,
    validation_path: Path,
    template_path: Path,
    december_path: Path,
    output_path: Path,
) -> None:
    bundle = PredictionBundle.load(bundle_path)
    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)

    predicted = bundle.predict(validation)
    rates = predicted.set_index("load_id")[bundle.prediction_col]
    submission = pd.read_csv(template_path)
    submission["predicted_rate"] = submission["load_id"].map(rates)
    submission.to_csv(output_path, index=False)

    december = pd.read_csv(december_path)
    model_rows = december.drop(columns="predicted_rate").copy()
    model_rows.insert(0, "load_id", [f"DEC-{index:06d}" for index in range(1, len(model_rows) + 1)])

    coordinates = _city_coordinates(train, validation)
    model_rows["pickup_lat"] = model_rows["pickup"].map(coordinates["lat"])
    model_rows["pickup_lon"] = model_rows["pickup"].map(coordinates["lon"])
    model_rows["delivery_lat"] = model_rows["delivery"].map(coordinates["lat"])
    model_rows["delivery_lon"] = model_rows["delivery"].map(coordinates["lon"])
    # The chart specification fixes every input except date. Use constant,
    # recent-data medians for model-required signals omitted by its template.
    model_rows["market_index"] = validation["market_index"].median()
    model_rows["quote_signal"] = validation["quote_signal"].median()

    december_prediction = bundle.predict(model_rows)
    december["predicted_rate"] = december_prediction[bundle.prediction_col].to_numpy()
    december.to_csv(december_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--train", type=Path, default=Path("data/raw/train-test.csv"))
    parser.add_argument("--validation", type=Path, default=Path("data/raw/validation.csv"))
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("data/raw/validation-predictions-template.csv"),
    )
    parser.add_argument(
        "--december", type=Path, default=Path("data/raw/december-chart-inputs.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("validation_predictions.csv"))
    args = parser.parse_args()
    generate(
        args.bundle,
        args.train,
        args.validation,
        args.template,
        args.december,
        args.output,
    )


if __name__ == "__main__":
    main()
