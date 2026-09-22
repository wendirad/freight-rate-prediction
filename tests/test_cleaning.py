import numpy as np
import pandas as pd
import pytest
from data.cleaning import (
    CategoryNormalizer,
    CleaningPipeline,
    DuplicateRemover,
    LaneDistanceOutlierRemover,
    MarketIndexImputer,
    TargetOutlierRemover,
    TypeCaster,
    WeightCleaner,
)


def test_cleaners_require_fit_before_transform() -> None:
    cleaner = TypeCaster()

    with pytest.raises(RuntimeError, match="TypeCaster must be fit before transform"):
        cleaner.transform(pd.DataFrame({"date": ["2025-01-01"]}))


def test_type_and_category_cleaners_transform_a_copy() -> None:
    data = pd.DataFrame(
        {
            "date": ["2025-01-01"],
            "distance": ["125.5"],
            "pickup": ["  Addis Ababa  "],
            "delivery": ["Djibouti "],
        }
    )
    original = data.copy(deep=True)
    pipeline = CleaningPipeline([TypeCaster(), CategoryNormalizer()])

    result = pipeline.fit_transform(data)

    pd.testing.assert_frame_equal(data, original)
    assert result.loc[0, "date"] == pd.Timestamp("2025-01-01")
    assert result.loc[0, "distance"] == pytest.approx(125.5)
    assert result.loc[0, "pickup"] == "Addis Ababa"
    assert result.loc[0, "delivery"] == "Djibouti"
    assert [
        (report.step, report.rows_before, report.rows_after)
        for report in pipeline.reports_
    ] == [
        ("TypeCaster", 1, 1),
        ("CategoryNormalizer", 1, 1),
    ]


def test_weight_cleaner_repairs_signs_and_uses_fitted_medians() -> None:
    training = pd.DataFrame(
        {
            "equipment": ["Dry", "Dry", "Flatbed", "Dry"],
            "weight": [-100.0, 300.0, 500.0, 0.0],
        }
    )
    cleaner = WeightCleaner().fit(training)
    prediction = pd.DataFrame(
        {
            "equipment": ["Dry", "Unknown", "Flatbed"],
            "weight": [np.nan, np.nan, -25.0],
        }
    )

    result = cleaner.transform(prediction)

    assert result["weight"].tolist() == pytest.approx([200.0, 300.0, 25.0])
    assert prediction["weight"].isna().tolist() == [True, True, False]
    assert prediction.loc[2, "weight"] == pytest.approx(-25.0)


def test_market_index_imputer_applies_each_fallback_in_order() -> None:
    training = pd.DataFrame(
        {
            "pickup": ["A", "A", "C", "E"],
            "delivery": ["B", "B", "D", "F"],
            "date": pd.to_datetime(
                ["2025-01-01", "2025-01-02", "2025-02-01", "2025-03-01"]
            ),
            "market_index": [10.0, 14.0, 30.0, 50.0],
        }
    )
    cleaner = MarketIndexImputer().fit(training)
    prediction = pd.DataFrame(
        {
            "pickup": ["X", "X", "A", "U", "U"],
            "delivery": ["Y", "Y", "B", "V", "V"],
            "date": pd.to_datetime(
                ["2025-04-01", "2025-04-01", "2025-04-02", "2025-02-02", "2025-04-02"]
            ),
            "market_index": [8.0, np.nan, np.nan, np.nan, np.nan],
        }
    )

    result = cleaner.transform(prediction)

    assert result["market_index"].tolist() == pytest.approx(
        [8.0, 8.0, 12.0, 30.0, 22.0]
    )


def test_duplicate_remover_only_drops_training_rows_and_reports_duplicates() -> None:
    data = pd.DataFrame(
        {
            "load_id": [1, 1, 1, 2],
            "distance": [10.0, 10.0, 20.0, 30.0],
        }
    )
    cleaner = DuplicateRemover().fit(data)

    prediction_result = cleaner.transform(data, is_training=False)
    training_result = cleaner.transform(data, is_training=True)

    pd.testing.assert_frame_equal(prediction_result, data)
    assert training_result["load_id"].tolist() == [1, 2]
    assert cleaner.last_report_ == {"exact_duplicates": 1, "duplicate_ids": 2}


def test_outlier_removers_preserve_prediction_rows() -> None:
    target_data = pd.DataFrame({"posted_rate": [10.0, 11.0, 12.0, 1_000.0]})
    target_cleaner = TargetOutlierRemover(lower_q=0.25, upper_q=0.75).fit(target_data)

    target_training = target_cleaner.transform(target_data, is_training=True)
    target_prediction = target_cleaner.transform(target_data, is_training=False)

    assert target_training["posted_rate"].tolist() == [11.0, 12.0]
    pd.testing.assert_frame_equal(target_prediction, target_data)

    distance_data = pd.DataFrame(
        {
            "pickup": ["A", "A", "A", "C"],
            "delivery": ["B", "B", "B", "D"],
            "distance": [100.0, 100.0, 300.0, 50.0],
        }
    )
    distance_cleaner = LaneDistanceOutlierRemover(tolerance=0.20).fit(distance_data)

    distance_training = distance_cleaner.transform(distance_data, is_training=True)
    distance_prediction = distance_cleaner.transform(distance_data, is_training=False)

    assert distance_training["distance"].tolist() == [100.0, 100.0, 50.0]
    pd.testing.assert_frame_equal(distance_prediction, distance_data)
