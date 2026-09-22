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


class CatFeaturesEstimator:
    """A RecordingEstimator whose fit also accepts cat_features, like
    CatBoostRegressor, so Trainer's cat_features passthrough can be tested.
    """

    def __init__(self, predictions: list[float] | None = None) -> None:
        self.predictions = np.asarray(predictions or [], dtype=float)
        self.fit_X: pd.DataFrame | None = None
        self.fit_cat_features: list[str] | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
        cat_features: list[str] | None = None,
    ) -> Any:
        self.fit_X = X.copy()
        self.fit_cat_features = cat_features
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
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


def test_trainer_rate_per_km_target_round_trip() -> None:
    data = pd.DataFrame(
        {"distance": [100.0, 200.0], "posted_rate": [500.0, 900.0]}
    )
    model = RecordingEstimator(predictions=[5.0, 4.5])
    trainer = Trainer(
        model,
        FeatureConfig(feature_cols=["distance"], target_transform="rate_per_km"),
    ).train(data)

    np.testing.assert_allclose(model.fit_y, [5.0, 4.5])
    np.testing.assert_allclose(trainer.predict(data), [500.0, 900.0])


def test_trainer_rate_per_km_rejects_non_positive_distance() -> None:
    data = pd.DataFrame({"distance": [0.0], "posted_rate": [500.0]})
    trainer = Trainer(
        RecordingEstimator(),
        FeatureConfig(feature_cols=["distance"], target_transform="rate_per_km"),
    )

    with pytest.raises(ValueError, match="positive distance"):
        trainer.train(data)


def test_trainer_log_rate_per_km_detrended_round_trip_uses_training_trend() -> None:
    train = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-11", "2025-01-21"]),
            "distance": [100.0, 100.0, 100.0],
            "posted_rate": np.exp([2.0, 2.1, 2.2]) * 100,
        }
    )
    validation = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-31"]),
            "distance": [200.0],
            "posted_rate": [1.0],
        }
    )
    model = RecordingEstimator(predictions=[0.0])
    trainer = Trainer(
        model,
        FeatureConfig(
            feature_cols=["distance"],
            target_transform="log_rate_per_km_detrended",
        ),
    ).train(train)

    np.testing.assert_allclose(model.fit_y, [0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(trainer.predict(validation), [np.exp(2.3) * 200])


def test_trainer_log_rate_per_km_detrended_rejects_non_positive_target() -> None:
    data = pd.DataFrame(
        {"date": ["2025-01-01"], "distance": [100.0], "posted_rate": [0.0]}
    )
    trainer = Trainer(
        RecordingEstimator(),
        FeatureConfig(
            feature_cols=["distance"],
            target_transform="log_rate_per_km_detrended",
        ),
    )

    with pytest.raises(ValueError, match="positive target"):
        trainer.train(data)


def test_trainer_passes_cat_features_when_estimator_supports_it() -> None:
    data = pd.DataFrame(
        {
            "distance": [100.0, 200.0],
            "lane": ["A -> B", "C -> D"],
            "posted_rate": [500.0, 900.0],
        }
    )
    model = CatFeaturesEstimator(predictions=[0.0, 0.0])
    trainer = Trainer(
        model,
        FeatureConfig(
            feature_cols=["distance", "lane"], categorical_cols=["lane"]
        ),
    )

    trainer.train(data)

    assert model.fit_cat_features == ["lane"]
    pd.testing.assert_frame_equal(model.fit_X, data[["distance", "lane"]])


def test_trainer_omits_cat_features_when_estimator_does_not_support_it() -> None:
    data = pd.DataFrame(
        {
            "distance": [100.0, 200.0],
            "posted_rate": [500.0, 900.0],
        }
    )
    model = RecordingEstimator(predictions=[0.0, 0.0])
    trainer = Trainer(
        model,
        FeatureConfig(feature_cols=["distance"], categorical_cols=["distance"]),
    )

    # RecordingEstimator.fit has no cat_features parameter; if Trainer tried
    # to pass one, this would raise a TypeError.
    trainer.train(data)

    assert trainer.is_fitted_


def test_trainer_filters_categorical_cols_to_those_in_feature_cols() -> None:
    data = pd.DataFrame(
        {
            "distance": [100.0, 200.0],
            "lane": ["A -> B", "C -> D"],
            "posted_rate": [500.0, 900.0],
        }
    )
    model = CatFeaturesEstimator(predictions=[0.0, 0.0])
    trainer = Trainer(
        model,
        FeatureConfig(
            feature_cols=["distance", "lane"],
            categorical_cols=["lane", "not_a_feature_col"],
        ),
    )

    trainer.train(data)

    assert model.fit_cat_features == ["lane"]


class _FakeCatBoostLikeEstimator:
    """Mimics enough of CatBoostRegressor's fit signature (eval_set,
    early_stopping_rounds, no eval_names) for Trainer to exercise the
    CatBoost-specific eval_set branch without needing real CatBoost.
    """

    def __init__(self) -> None:
        self.fit_eval_set: list | None = None
        self.fit_early_stopping_rounds: int | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
        eval_set: list | None = None,
        early_stopping_rounds: int | None = None,
    ) -> Any:
        self.fit_eval_set = eval_set
        self.fit_early_stopping_rounds = early_stopping_rounds
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))


_FakeCatBoostLikeEstimator.__module__ = "catboost.core"


def test_trainer_passes_only_validation_pair_and_native_early_stopping_for_catboost() -> (
    None
):
    """CatBoost auto-tracks its own training ("learn") curve; passing the
    training pair again inside eval_set (the LightGBM convention) would add
    a spurious extra evals_result_ key that shadows the real validation
    curve. Early stopping must go through the native early_stopping_rounds
    kwarg, not a lgb.early_stopping callback.
    """
    train_data = pd.DataFrame({"distance": [100.0, 200.0], "posted_rate": [500.0, 900.0]})
    val_data = pd.DataFrame({"distance": [150.0], "posted_rate": [700.0]})
    model = _FakeCatBoostLikeEstimator()
    trainer = Trainer(model, FeatureConfig(feature_cols=["distance"]))

    trainer.train(train_data, eval_df=val_data, early_stopping_rounds=50)

    assert model.fit_eval_set is not None
    assert len(model.fit_eval_set) == 1
    val_X, val_y = model.fit_eval_set[0]
    pd.testing.assert_frame_equal(val_X, val_data[["distance"]])
    pd.testing.assert_series_equal(val_y, val_data["posted_rate"])
    assert model.fit_early_stopping_rounds == 50


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
