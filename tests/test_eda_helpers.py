from pathlib import Path

import marimo as mo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from utils.eda_helpers import (
    EDAConfig,
    cramers_v,
    find_cluster_driver,
    haversine,
    load_csv,
    load_freight_data,
    plot_qq,
    recover_market_index,
    report_findings,
    within_std_proportions,
)


def test_eda_config_rejects_missing_dataset(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match=str(missing_path)):
        EDAConfig(train_test_path=missing_path)


def test_load_csv_parses_requested_date_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame({"date": ["2025-01-01", "2025-01-02"], "value": [1, 2]}).to_csv(
        csv_path, index=False
    )

    result = load_csv(str(csv_path), date_cols=("date",))

    assert pd.api.types.is_datetime64_any_dtype(result["date"])
    assert result["value"].tolist() == [1, 2]


def test_load_freight_data_uses_config_and_validates_width(tmp_path: Path) -> None:
    csv_path = tmp_path / "freight.csv"
    expected = pd.DataFrame({"date": ["2025-01-01"], "posted_rate": [1_000.0]})
    expected.to_csv(csv_path, index=False)
    config = EDAConfig(
        train_test_path=csv_path,
        date_cols=("date",),
        total_cols=2,
    )

    result = load_freight_data(config)

    assert result.shape == (1, 2)
    assert result.loc[0, "date"] == pd.Timestamp("2025-01-01")

    invalid_config = EDAConfig(
        train_test_path=csv_path,
        date_cols=("date",),
        total_cols=3,
    )
    with pytest.raises(ValueError, match="Expected 3 columns, found 2"):
        load_freight_data(invalid_config)


def test_plot_qq_returns_configured_figure_and_ignores_missing_values() -> None:
    reference = plt.figure(figsize=(6, 2), dpi=100)
    figure = plot_qq([1.0, 2.0, np.nan, 3.0], figsize=(6, 2), dpi=100)

    try:
        assert isinstance(figure, plt.Figure)
        assert tuple(figure.get_size_inches()) == pytest.approx((6, 2))
        assert figure.dpi == pytest.approx(reference.dpi)
        assert len(figure.axes) == 1
        assert len(figure.axes[0].lines) == 2
    finally:
        plt.close(reference)
        plt.close(figure)


def test_within_std_proportions_returns_empirical_and_normal_comparison() -> None:
    result = within_std_proportions([-1.0, 0.0, 1.0, np.nan], n=2)
    theoretical = np.array([stats.norm.cdf(k) - stats.norm.cdf(-k) for k in (1, 2)])

    assert result["Standard Deviations (k)"].tolist() == ["±1 SD", "±2 SD"]
    np.testing.assert_allclose(result["Empirical Proportion"], [1 / 3, 1.0])
    np.testing.assert_allclose(result["Offset"], np.array([1 / 3, 1.0]) - theoretical)


def test_report_findings_returns_marimo_html() -> None:
    result = report_findings("A useful result", title="Conclusion", kind="success")

    assert isinstance(result, mo.Html)
    assert "A useful result" in result.text
    assert "Conclusion" in result.text
    assert "success" in result.text


def test_cramers_v_distinguishes_independent_and_associated_tables() -> None:
    independent = pd.DataFrame([[5, 5], [5, 5]])
    associated = pd.DataFrame([[10, 0], [0, 10]])

    assert cramers_v(independent) == pytest.approx(0.0)
    assert cramers_v(associated) > 0.8


def test_cramers_v_returns_nan_when_association_is_undefined() -> None:
    one_category = pd.DataFrame([[5, 5]])

    assert np.isnan(cramers_v(one_category))


def test_haversine_supports_scalar_and_vector_inputs() -> None:
    assert haversine(40.7128, -74.0060, 40.7128, -74.0060) == pytest.approx(0.0)
    assert haversine(40.7128, -74.0060, 42.3601, -71.0589) == pytest.approx(
        306.1, abs=0.2
    )

    result = haversine(
        np.array([40.7128, 40.7128]),
        np.array([-74.0060, -74.0060]),
        np.array([40.7128, 42.3601]),
        np.array([-74.0060, -71.0589]),
    )
    np.testing.assert_allclose(result, [0.0, 306.1], atol=0.2)


def test_find_cluster_driver_ranks_valid_candidates_and_preserves_rows() -> None:
    values = np.concatenate([np.linspace(0.0, 0.9, 10), np.linspace(10.0, 10.9, 10)])
    data = pd.DataFrame(
        {
            "value": np.append(values, np.nan),
            "aligned": ["low"] * 10 + ["high"] * 10 + ["missing"],
            "mixed": (["even", "odd"] * 10) + ["missing"],
            "constant": ["same"] * 21,
        },
        index=np.arange(100, 121),
    )

    results, clustered = find_cluster_driver(
        data,
        "value",
        ["aligned", "mixed", "constant", "not_present"],
        n_components=2,
        random_state=7,
    )

    assert results["column"].tolist() == ["aligned", "mixed"]
    assert results.iloc[0]["cramers_v"] > results.iloc[1]["cramers_v"]
    assert results["n_categories"].tolist() == [2, 2]
    assert clustered.index.tolist() == list(range(100, 120))
    assert set(clustered["_cluster"].unique()) == {0, 1}


def test_recover_market_index_uses_lane_date_median_without_mutating_input() -> None:
    date = pd.Timestamp("2025-01-01")
    data = pd.DataFrame(
        {
            "pickup": ["A", "A", "A", "X", "X"],
            "delivery": ["B", "B", "B", "Y", "Z"],
            "date": [date] * 5,
            "market_index": [1.0, 1.4, np.nan, 2.0, np.nan],
        },
        index=[10, 11, 12, 13, 14],
    )
    original = data.copy(deep=True)

    recovered, recovered_values = recover_market_index(data)

    pd.testing.assert_frame_equal(data, original)
    assert recovered.loc[12, "market_index"] == pytest.approx(1.2)
    assert np.isnan(recovered.loc[14, "market_index"])
    assert recovered_values.index.tolist() == [12, 14]
    assert recovered_values.loc[12] == pytest.approx(1.2)
    assert np.isnan(recovered_values.loc[14])
