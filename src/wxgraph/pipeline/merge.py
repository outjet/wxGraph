"""Merge utilities for combining multiple model datasets."""

from __future__ import annotations

from typing import Mapping

import pandas as pd


def merge_models(dfs: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge point-dataframes from multiple models and mark the originating model.
    """

    if not dfs:
        return pd.DataFrame()

    pieces: list[pd.DataFrame] = []
    for model_name, df in dfs.items():
        piece = df.copy()
        piece["source_model"] = model_name
        pieces.append(piece)

    merged = pd.concat(pieces, ignore_index=True, sort=False)
    if "valid_time" in merged:
        merged = merged.sort_values("valid_time").reset_index(drop=True)
    return merged
