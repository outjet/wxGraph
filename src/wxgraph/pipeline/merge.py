"""Merge utilities for combining multiple model datasets."""

from __future__ import annotations

from typing import Mapping

import pandas as pd


def merge_models(dfs: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge point-dataframes from multiple models along the columns they share.
    """

    if not dfs:
        return pd.DataFrame()
    concatenated = pd.concat(dfs.values(), ignore_index=True, sort=False)
    concatenated["source_model"] = pd.Series(
        [model for model, df in dfs.items() for _ in range(len(df))], dtype="string"
    )
    return concatenated
