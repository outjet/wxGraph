"""Normalization helpers for point-data frames."""

from __future__ import annotations

import pandas as pd


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename and unit-convert columns from a raw extraction into a canonical set.
    """

    normalized = df.copy()
    if "temp_k" in normalized:
        normalized["temp_c"] = normalized["temp_k"] - 273.15
        normalized["temp_f"] = normalized["temp_c"] * 9.0 / 5.0 + 32.0
    return normalized
