import pandas as pd
from utils.caching import cached_dataframe_call, config_hash


def test_second_call_with_same_key_does_not_recompute(tmp_path) -> None:
    calls = {"n": 0}

    def compute() -> pd.DataFrame:
        calls["n"] += 1
        return pd.DataFrame({"a": [1, 2, 3]})

    key = config_hash({"step": "clean", "param": 1})

    first = cached_dataframe_call(compute, key, tmp_path)
    second = cached_dataframe_call(compute, key, tmp_path)

    assert calls["n"] == 1
    pd.testing.assert_frame_equal(first, second)


def test_different_keys_do_recompute(tmp_path) -> None:
    calls = {"n": 0}

    def compute() -> pd.DataFrame:
        calls["n"] += 1
        return pd.DataFrame({"a": [calls["n"]]})

    cached_dataframe_call(compute, config_hash({"param": 1}), tmp_path)
    cached_dataframe_call(compute, config_hash({"param": 2}), tmp_path)

    assert calls["n"] == 2
