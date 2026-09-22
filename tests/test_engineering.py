from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from features.engineering import (
    CyclicalMonthEncoder,
    DayOfWeekEncoder,
    EquipmentEncoder,
    FeatureEngineeringPipeline,
    FreightDomainFeatures,
    LaneBuilder,
    MarketIndexEMA,
    QuoteSignalZScore,
    build_default_feature_pipeline,
)


@pytest.mark.parametrize(
    "transformer",
    [
        LaneBuilder(),
        CyclicalMonthEncoder(),
        DayOfWeekEncoder(),
        EquipmentEncoder(),
        MarketIndexEMA(),
        QuoteSignalZScore(),
        FreightDomainFeatures(),
    ],
)
def test_feature_transformers_require_fit_before_transform(transformer) -> None:
    with pytest.raises(
        RuntimeError,
        match=rf"{transformer.__class__.__name__} must be fit before transform",
    ):
        transformer.transform(pd.DataFrame())


def test_lane_and_date_encoders_add_expected_features_without_mutating_input() -> None:
    data = pd.DataFrame(
        {
            "pickup": ["Addis Ababa", "Dire Dawa"],
            "delivery": ["Djibouti", "Hawassa"],
            "date": ["2025-01-06", "2025-04-13"],
        }
    )
    original = data.copy(deep=True)
    pipeline = FeatureEngineeringPipeline(
        [LaneBuilder(), CyclicalMonthEncoder(), DayOfWeekEncoder()]
    )

    result = pipeline.fit_transform(data)

    pd.testing.assert_frame_equal(data, original)
    assert result["lane"].tolist() == [
        "Addis Ababa -> Djibouti",
        "Dire Dawa -> Hawassa",
    ]
    np.testing.assert_allclose(result["month_sin"], [0.5, np.sqrt(3) / 2])
    np.testing.assert_allclose(result["month_cos"], [np.sqrt(3) / 2, -0.5])
    assert result["day_of_week"].tolist() == [0, 6]


def test_equipment_encoder_reuses_sorted_training_categories() -> None:
    training = pd.DataFrame({"equipment": ["Reefer", "Dry Van", "Reefer", np.nan]})
    encoder = EquipmentEncoder().fit(training)
    prediction = pd.DataFrame(
        {"equipment": ["Dry Van", "Flatbed", np.nan]}, index=[4, 2, 9]
    )
    original = prediction.copy(deep=True)

    result = encoder.transform(prediction)

    pd.testing.assert_frame_equal(prediction, original)
    assert encoder.categories_ == ["Dry Van", "Reefer"]
    assert result.index.tolist() == [4, 2, 9]
    assert result["equipment_Dry Van"].tolist() == [1, 0, 0]
    assert result["equipment_Reefer"].tolist() == [0, 0, 0]
    assert "equipment_Flatbed" not in result


def test_market_index_ema_prediction_blends_value_with_fitted_lane_state_only() -> None:
    training = pd.DataFrame(
        {
            "lane": ["A -> B", "A -> B", "C -> D"],
            "date": pd.to_datetime(["2025-01-02", "2025-01-01", "2025-01-01"]),
            "market_index": [30.0, 10.0, 20.0],
        }
    )
    encoder = MarketIndexEMA(span=3).fit(training)
    prediction = pd.DataFrame(
        {
            "lane": ["A -> B", "A -> B", "unseen"],
            "date": pd.to_datetime(["2025-01-04", "2025-01-03", "2025-01-02"]),
            "market_index": [50.0, 40.0, 10.0],
        },
        index=[5, 3, 8],
    )
    original = prediction.copy(deep=True)

    result = encoder.transform(prediction)

    pd.testing.assert_frame_equal(prediction, original)
    assert encoder.global_fallback_ == pytest.approx(20.0)
    assert encoder.lane_last_ema_ == pytest.approx({"A -> B": 20.0, "C -> D": 20.0})
    assert result.index.tolist() == [5, 3, 8]
    # Each row is blended only against fitted training state (alpha=0.5):
    # idx5: 0.5*50 + 0.5*20 = 35.0; idx3: 0.5*40 + 0.5*20 = 30.0;
    # idx8 (unseen lane): falls back to global_fallback_ for "prev",
    # 0.5*10 + 0.5*20 = 15.0. No row's output depends on another row.
    assert result["market_index_ema"].tolist() == pytest.approx([35.0, 30.0, 15.0])
    # Fitted state is untouched by transforming prediction data.
    assert encoder.lane_last_ema_ == pytest.approx({"A -> B": 20.0, "C -> D": 20.0})


def test_market_index_ema_prediction_transform_is_batch_invariant() -> None:
    training = pd.DataFrame(
        {
            "lane": ["A -> B", "A -> B", "C -> D"],
            "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-01"]),
            "market_index": [10.0, 30.0, 20.0],
        }
    )
    encoder = MarketIndexEMA(span=3).fit(training)
    fitted_state_before = dict(encoder.lane_last_ema_)

    target_row = pd.DataFrame(
        {
            "lane": ["A -> B"],
            "date": pd.to_datetime(["2025-01-04"]),
            "market_index": [50.0],
        },
        index=[0],
    )
    adversarial_row = pd.DataFrame(
        {
            "lane": ["A -> B"],
            "date": pd.to_datetime(["2025-01-03"]),
            "market_index": [999.0],
        },
        index=[1],
    )
    batch = pd.concat([target_row, adversarial_row])
    batch_reordered = pd.concat([adversarial_row, target_row])

    alone_result = encoder.transform(target_row, is_training=False)
    batch_result = encoder.transform(batch, is_training=False)
    reordered_result = encoder.transform(batch_reordered, is_training=False)

    target_alone = alone_result.loc[0, "market_index_ema"]
    target_in_batch = batch_result.loc[0, "market_index_ema"]
    target_in_reordered = reordered_result.loc[0, "market_index_ema"]

    assert target_alone == pytest.approx(target_in_batch)
    assert target_alone == pytest.approx(target_in_reordered)
    assert batch_result.loc[1, "market_index_ema"] == pytest.approx(
        reordered_result.loc[1, "market_index_ema"]
    )
    # Transforming prediction data never mutates fitted training state.
    assert encoder.lane_last_ema_ == pytest.approx(fitted_state_before)


def test_market_index_ema_training_transform_does_not_use_fitted_lane_state() -> None:
    training = pd.DataFrame(
        {
            "lane": ["A -> B", "A -> B"],
            "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "market_index": [10.0, 30.0],
        }
    )
    encoder = MarketIndexEMA(span=3).fit(training)

    result = encoder.transform(training, is_training=True)

    assert result["market_index_ema"].tolist() == pytest.approx([15.0, 22.5])


def test_quote_signal_zscore_uses_training_statistics() -> None:
    training = pd.DataFrame({"quote_signal": [1.0, 2.0, 3.0]})
    encoder = QuoteSignalZScore().fit(training)
    prediction = pd.DataFrame({"quote_signal": [0.0, 2.0, 4.0]})

    result = encoder.transform(prediction)

    assert encoder.mean_ == pytest.approx(2.0)
    assert encoder.std_ == pytest.approx(1.0)
    assert result["quote_signal_zscore"].tolist() == pytest.approx([-2.0, 0.0, 2.0])


def test_freight_domain_features_add_interactions_with_fitted_time_origin() -> None:
    training = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-11"]),
            "distance": [100.0, 200.0],
            "weight": [1000.0, 3000.0],
            "market_index": [2.0, 3.0],
            "quote_signal": [0.5, 0.25],
            "pickup_lat": [0.0, 10.0],
            "pickup_lon": [0.0, 10.0],
            "delivery_lat": [1.0, 12.0],
            "delivery_lon": [1.0, 13.0],
        }
    )
    validation = training.iloc[[1]].copy()
    validation["date"] = pd.to_datetime(["2025-01-21"])

    transformer = FreightDomainFeatures().fit(training)
    result = transformer.transform(validation)

    assert result["weight_per_km"].iloc[0] == pytest.approx(15.0)
    assert result["market_distance"].iloc[0] == pytest.approx(600.0)
    assert result["quote_distance"].iloc[0] == pytest.approx(50.0)
    assert result["days_since_training_start"].iloc[0] == pytest.approx(20.0)
    assert result["geo_distance_km"].iloc[0] > 0
    assert np.isfinite(result["route_distance_ratio"].iloc[0])


def test_default_pipeline_composes_all_features_and_round_trips(
    tmp_path: Path,
) -> None:
    training = pd.DataFrame(
        {
            "pickup": ["A", "C"],
            "delivery": ["B", "D"],
            "date": pd.to_datetime(["2025-01-06", "2025-02-11"]),
            "equipment": ["Dry Van", "Reefer"],
            "market_index": [10.0, 30.0],
            "quote_signal": [1.0, 3.0],
        }
    )
    pipeline = build_default_feature_pipeline()
    result = pipeline.fit_transform(training)
    path = tmp_path / "feature-pipeline.joblib"

    pipeline.save(str(path))
    restored = FeatureEngineeringPipeline.load(str(path))
    restored_result = restored.transform(training, is_training=True)

    expected_columns = {
        "lane",
        "month_sin",
        "month_cos",
        "day_of_week",
        "equipment_Dry Van",
        "equipment_Reefer",
        "market_index_ema",
        "quote_signal_zscore",
    }
    assert expected_columns <= set(result.columns)
    pd.testing.assert_frame_equal(restored_result, result)
