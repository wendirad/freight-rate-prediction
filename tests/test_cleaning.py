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

    # Row 1 (X, Y, 2025-04-01) shares pickup/delivery/date with row 0's
    # known 8.0, but that lane-date combination was never seen in
    # training, so stage one (a fitted training lookup) does not fill it
    # from row 0 anymore; it falls through to the global training median
    # (22.0) instead. Row 2 (A, B) hits the fitted lane median (12.0).
    # Row 3 (U, V, 2025-02-02) hits the fitted month-2 median (30.0).
    # Row 4 (U, V, 2025-04-02) falls through to the global median (22.0).
    assert result["market_index"].tolist() == pytest.approx(
        [8.0, 22.0, 12.0, 30.0, 22.0]
    )


def test_market_index_imputer_transform_is_batch_invariant() -> None:
    """The exact bug this guards against: stage one used to compute a
    lane/date median from whatever rows happened to be in the dataframe
    being transformed, so one row's imputed value depended on which other
    rows were in the same validation batch. It must now come only from
    fit().
    """
    training = pd.DataFrame(
        {
            "pickup": ["A"],
            "delivery": ["B"],
            "date": pd.to_datetime(["2025-01-01"]),
            "market_index": [10.0],
        }
    )
    cleaner = MarketIndexImputer().fit(training)

    target_row = pd.DataFrame(
        {
            "pickup": ["X"],
            "delivery": ["Y"],
            "date": pd.to_datetime(["2025-04-01"]),
            "market_index": [np.nan],
        },
        index=[0],
    )
    adversarial_row = pd.DataFrame(
        {
            "pickup": ["X"],
            "delivery": ["Y"],
            "date": pd.to_datetime(["2025-04-01"]),
            "market_index": [999.0],
        },
        index=[1],
    )
    batch = pd.concat([target_row, adversarial_row])
    batch_reordered = pd.concat([adversarial_row, target_row])

    alone_result = cleaner.transform(target_row)
    batch_result = cleaner.transform(batch)
    reordered_result = cleaner.transform(batch_reordered)

    target_alone = alone_result.loc[0, "market_index"]
    target_in_batch = batch_result.loc[0, "market_index"]
    target_in_reordered = reordered_result.loc[0, "market_index"]

    assert target_alone == pytest.approx(target_in_batch)
    assert target_alone == pytest.approx(target_in_reordered)
    assert target_alone != pytest.approx(999.0)
    assert batch_result.loc[1, "market_index"] == pytest.approx(
        reordered_result.loc[1, "market_index"]
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
