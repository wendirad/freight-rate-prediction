"""Generic dataframe caching keyed by config hash.

Separate from CleaningPipeline.save/load and FeatureEngineeringPipeline.save/
load, which persist the fitted pipeline object. In the experiment path
(run_experiment.py), this caches only the loaded, schema-validated,
chronologically sorted raw source dataframe: cleaning and feature
engineering happen per cross-validation fold on data that must not be fit
before entering the fold loop, so their outputs are never cached here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def config_hash(config: dict[str, Any]) -> str:
    """Stable hash of a config dict, used as a cache key component."""
    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def cached_dataframe_call(
    compute_fn,
    cache_key: str,
    cache_dir: str | Path,
) -> pd.DataFrame:
    """Returns the parquet cached at cache_dir/cache_key.parquet if present,
    otherwise calls compute_fn(), writes its result to that path, and
    returns it.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_key}.parquet"

    if cache_path.exists():
        return pd.read_parquet(cache_path)

    df = compute_fn()
    df.to_parquet(cache_path)
    return df
