from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from serving.inference import PredictionBundle


class CopyTransformer:
    def transform(self, frame: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        return frame.copy()


class DistancePredictor:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["distance"].to_numpy(dtype=float) * 2


def make_bundle() -> PredictionBundle:
    return PredictionBundle(
        cleaner=CopyTransformer(),
        feature_pipeline=CopyTransformer(),
        trainer=DistancePredictor(),
        input_columns=["load_id", "distance"],
    )


def test_prediction_bundle_preserves_input_and_appends_predictions() -> None:
    source = pd.DataFrame(
        {"load_id": ["a", "b"], "distance": [100.0, 250.0]},
        index=[5, 2],
    )
    original = source.copy(deep=True)

    result = make_bundle().predict(source)

    pd.testing.assert_frame_equal(source, original)
    assert result.index.tolist() == [5, 2]
    assert result["predicted_posted_rate"].tolist() == [200.0, 500.0]


def test_prediction_bundle_reports_all_missing_columns() -> None:
    with pytest.raises(
        ValueError,
        match=r"missing required columns: \['load_id', 'distance'\]",
    ):
        make_bundle().predict(pd.DataFrame({"other": [1]}))


def test_prediction_bundle_round_trips_through_joblib(tmp_path) -> None:
    path = tmp_path / "model.joblib"
    bundle = make_bundle()

    bundle.save(path)
    restored = PredictionBundle.load(path)
    result = restored.predict(pd.DataFrame({"load_id": ["a"], "distance": [12.5]}))

    assert result["predicted_posted_rate"].tolist() == [25.0]
