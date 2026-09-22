from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from features.engineering import (
    CyclicalMonthEncoder,
    DayOfWeekEncoder,
    EquipmentEncoder,
    FeatureEngineeringPipeline,
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


def test_market_index_ema_uses_fitted_lane_state_and_chronological_order() -> None:
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
    assert result["market_index_ema"].tolist() == pytest.approx([40.0, 30.0, 15.0])


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
