import pandas as pd
import pytest
from data.folds import make_expanding_folds


def _dated_df(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {"date": pd.date_range("2025-01-01", periods=n, freq="D"), "value": range(n)}
    )


def test_folds_are_strictly_increasing_and_contiguous() -> None:
    df = _dated_df(30)
    folds = make_expanding_folds(df, n_folds=4)

    assert len(folds) == 4
    for train_idx, val_idx in folds:
        assert train_idx.max() < val_idx.min()
        assert list(train_idx) == sorted(train_idx)
        assert list(val_idx) == sorted(val_idx)


def test_folds_do_not_overlap_in_time_across_the_sequence() -> None:
    df = _dated_df(30)
    folds = make_expanding_folds(df, n_folds=4)

    for i in range(len(folds) - 1):
        _, val_i = folds[i]
        train_next, _ = folds[i + 1]
        assert val_i.max() <= train_next.max()
        assert set(val_i).issubset(set(train_next))

    for train_idx, val_idx in folds:
        assert set(train_idx).isdisjoint(set(val_idx))


def test_expanding_window_grows_each_fold() -> None:
    df = _dated_df(30)
    folds = make_expanding_folds(df, n_folds=4)

    train_sizes = [len(train_idx) for train_idx, _ in folds]
    assert train_sizes == sorted(train_sizes)
    assert len(set(train_sizes)) == len(train_sizes)


def test_raises_when_not_enough_rows_for_requested_folds() -> None:
    df = _dated_df(3)
    with pytest.raises(ValueError, match="not enough rows"):
        make_expanding_folds(df, n_folds=5)
