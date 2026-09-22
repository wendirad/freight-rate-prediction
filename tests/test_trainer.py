from typing import Any

import numpy as np
import pandas as pd
import pytest
from models.trainer import (
    FeatureConfig,
    Trainer,
    build_equipment_sample_weights,
    compute_grouped_metrics,
    compute_metrics,
)


class RecordingEstimator:
    def __init__(self, predictions: list[float] | None = None) -> None:
        self.predictions = np.asarray(predictions or [], dtype=float)
        self.fit_X: pd.DataFrame | None = None
        self.fit_y: pd.Series | None = None
        self.fit_sample_weight: np.ndarray | None = None
        self.predict_X: pd.DataFrame | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> Any:
        self.fit_X = X.copy()
        self.fit_y = y.copy()
        self.fit_sample_weight = sample_weight
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.predict_X = X.copy()
        return self.predictions.copy()


def test_compute_metrics_and_clips_negative_predictions() -> None:
    y_true = np.array([1.0, 3.0])
    y_pred = np.array([-2.0, 5.0])

    result = compute_metrics(y_true, y_pred)

    expected_rmsle = np.sqrt(
        ((np.log1p(1.0) - np.log1p(0.0)) ** 2 + (np.log1p(3.0) - np.log1p(5.0)) ** 2)
        / 2
    )
    assert result == pytest.approx(
        {
            "mae": 2.5,
            "rmse": np.sqrt(6.5),
            "rmsle": expected_rmsle,
            "mape": (3.0 + 2.0 / 3.0) / 2 * 100,
        }
    )


def test_compute_grouped_metrics_calculates_each_group_and_count() -> None:
    data = pd.DataFrame(
        {
            "equipment": ["Dry Van", "Dry Van", "Reefer"],
            "actual": [100.0, 200.0, 300.0],
            "prediction": [90.0, 220.0, 330.0],
        }
    )

    result = compute_grouped_metrics(
        data,
        y_true_col="actual",
        y_pred_col="prediction",
        group_col="equipment",
    )

    assert set(result) == {"Dry Van", "Reefer"}
    assert result["Dry Van"]["n"] == 2
    assert result["Dry Van"]["mae"] == pytest.approx(15.0)
    assert result["Reefer"]["n"] == 1
    assert result["Reefer"]["mae"] == pytest.approx(30.0)


def test_trainer_requires_training_before_prediction() -> None:
    trainer = Trainer(RecordingEstimator(), FeatureConfig(feature_cols=["distance"]))

    with pytest.raises(RuntimeError, match="model must be trained before predict"):
        trainer.predict(pd.DataFrame({"distance": [100.0]}))


def test_trainer_selects_configured_columns_and_forwards_sample_weights() -> None:
    data = pd.DataFrame(
        {
            "distance": [100.0, 200.0],
            "market_index": [1.1, 1.3],
            "ignored": [10, 20],
            "posted_rate": [500.0, 900.0],
            "row_weight": [0.25, 1.0],
        },
        index=[4, 7],
    )
    model = RecordingEstimator(predictions=[510.0, 880.0])
    trainer = Trainer(
        model,
        FeatureConfig(
            feature_cols=["market_index", "distance"], weight_col="row_weight"
        ),
    )

    returned = trainer.train(data)
    predictions = trainer.predict(data)

    assert returned is trainer
    assert trainer.is_fitted_
    pd.testing.assert_frame_equal(model.fit_X, data[["market_index", "distance"]])
    pd.testing.assert_series_equal(model.fit_y, data["posted_rate"])
    np.testing.assert_array_equal(model.fit_sample_weight, [0.25, 1.0])
    pd.testing.assert_frame_equal(model.predict_X, data[["market_index", "distance"]])
    np.testing.assert_array_equal(predictions, [510.0, 880.0])


def test_validate_returns_grouped_metrics_without_mutation() -> None:
    data = pd.DataFrame(
        {
            "distance": [100.0, 200.0, 300.0],
            "posted_rate": [100.0, 200.0, 300.0],
            "equipment": ["Dry Van", "Dry Van", "Reefer"],
        }
    )
    original = data.copy(deep=True)
    trainer = Trainer(
        RecordingEstimator(predictions=[90.0, 220.0, 330.0]),
        FeatureConfig(feature_cols=["distance"]),
    ).train(data)

    result = trainer.validate(data)

    pd.testing.assert_frame_equal(data, original)
    assert result["overall"]["mae"] == pytest.approx(20.0)
    assert result["by_group"]["Dry Van"]["n"] == 2
    assert result["by_group"]["Dry Van"]["mae"] == pytest.approx(15.0)
    assert result["by_group"]["Reefer"]["n"] == 1
    assert result["by_group"]["Reefer"]["mae"] == pytest.approx(30.0)


def test_trainer_validate_skips_grouping_when_group_column_is_unavailable() -> None:
    data = pd.DataFrame({"distance": [100.0], "posted_rate": [500.0]})
    trainer = Trainer(
        RecordingEstimator(predictions=[490.0]),
        FeatureConfig(feature_cols=["distance"], group_col="equipment"),
    ).train(data)

    result = trainer.validate(data)

    assert result["by_group"] == {}


def test_equipment_weights_balance_groups_and_keep_order() -> None:
    data = pd.DataFrame(
        {"equipment": ["Dry Van", "Reefer", "Dry Van", "Dry Van"]},
        index=[8, 2, 5, 1],
    )

    result = build_equipment_sample_weights(data)

    np.testing.assert_allclose(result, [1 / 3, 1.0, 1 / 3, 1 / 3])
