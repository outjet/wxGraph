"""Normalization helpers for point-data frames."""

from __future__ import annotations

import pandas as pd


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize basic meteorological units and timestamps.
    """

    normalized = df.copy()
    if "valid_time" in normalized:
        normalized["valid_time"] = pd.to_datetime(normalized["valid_time"])

    if "temp_k" in normalized and "temp_c" not in normalized:
        normalized["temp_c"] = normalized["temp_k"] - 273.15
    if "temp_c" in normalized:
        normalized["temp_k"] = normalized.get("temp_k", normalized["temp_c"] + 273.15)
        if "temp_f" not in normalized:
            normalized["temp_f"] = normalized["temp_c"] * 9.0 / 5.0 + 32.0

    if "wind10m_ms" in normalized and "wind10m_mph" not in normalized:
        normalized["wind10m_mph"] = normalized["wind10m_ms"] * 2.23694
    if "wind10m_mph" in normalized and "wind10m_ms" not in normalized:
        normalized["wind10m_ms"] = normalized["wind10m_mph"] / 2.23694

    if "rh" in normalized and "rh_percent" not in normalized:
        normalized["rh_percent"] = normalized["rh"]

    normalized["model"] = normalized.get("model", "unknown")
    return normalized
