from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.evaluator import EvaluationReport, Evaluator, compare_runs
from models.trainer import FeatureConfig, Trainer


class FixedEstimator:
    def __init__(self, predictions: list[float]) -> None:
        self.predictions = np.asarray(predictions, dtype=float)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> FixedEstimator:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predictions.copy()


def test_evaluate_builds_metrics_residuals_and_ranked_worst_predictions() -> None:
    data = pd.DataFrame(
        {
            "load_id": [101, 102, 103],
            "equipment": ["Dry Van", "Dry Van", "Reefer"],
            "distance": [100.0, 200.0, 300.0],
            "posted_rate": [100.0, 200.0, 300.0],
        },
        index=[8, 3, 5],
    )
    original = data.copy(deep=True)
    predictions = [130.0, 190.0, 240.0]
    trainer = Trainer(
        FixedEstimator(predictions),
        FeatureConfig(feature_cols=["distance"]),
    ).train(data)
    evaluator = Evaluator(
        trainer,
        id_cols=["load_id", "missing_column", "equipment"],
    )

    report = evaluator.evaluate(data, n_worst=2)

    pd.testing.assert_frame_equal(data, original)
    assert report.overall["mae"] == pytest.approx(100 / 3)
    assert report.by_group["Dry Van"]["n"] == 2
    assert report.by_group["Dry Van"]["mae"] == pytest.approx(20.0)
    assert report.by_group["Reefer"]["n"] == 1
    assert report.by_group["Reefer"]["mae"] == pytest.approx(60.0)

    residuals = np.array([-30.0, 10.0, 60.0])
    assert report.residual_stats == pytest.approx(
        {
            "mean": residuals.mean(),
            "std": residuals.std(),
            "skew": pd.Series(residuals).skew(),
        }
    )
    assert report.worst_predictions.columns.tolist() == [
        "load_id",
        "equipment",
        "posted_rate",
        "prediction",
        "abs_error",
    ]
    assert report.worst_predictions["load_id"].tolist() == [103, 101]
    assert report.worst_predictions["prediction"].tolist() == [240.0, 130.0]
    assert report.worst_predictions["abs_error"].tolist() == [60.0, 30.0]
    assert report.worst_predictions.index.tolist() == [0, 1]


def test_evaluate_skips_grouping_when_disabled() -> None:
    data = pd.DataFrame(
        {
            "distance": [100.0, 200.0, 300.0],
            "posted_rate": [110.0, 190.0, 310.0],
        }
    )
    trainer = Trainer(
        FixedEstimator([100.0, 200.0, 300.0]),
        FeatureConfig(feature_cols=["distance"], group_col=None),
    ).train(data)

    report = Evaluator(trainer).evaluate(data)

    assert report.by_group == {}
    assert report.worst_predictions.columns.tolist() == [
        "distance",
        "posted_rate",
        "prediction",
        "abs_error",
    ]


def test_compare_runs_sorts_reports_by_mae_and_preserves_metrics() -> None:
    empty_worst = pd.DataFrame()
    reports = {
        "baseline": EvaluationReport(
            overall={"mae": 25.0, "rmse": 30.0},
            by_group={},
            residual_stats={},
            worst_predictions=empty_worst,
        ),
        "candidate": EvaluationReport(
            overall={"mae": 10.0, "rmse": 14.0},
            by_group={},
            residual_stats={},
            worst_predictions=empty_worst,
        ),
    }

    result = compare_runs(reports)

    expected = pd.DataFrame(
        {
            "run": ["candidate", "baseline"],
            "mae": [10.0, 25.0],
            "rmse": [14.0, 30.0],
        }
    )
    pd.testing.assert_frame_equal(result, expected)
