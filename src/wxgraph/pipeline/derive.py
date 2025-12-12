"""Derived field calculators for point forecasts."""

from __future__ import annotations

import pandas as pd


def add_dewpoint(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute dewpoint values from temperature and humidity inputs.
    """

    result = df.copy()
    if "temp_c" in result and "rh_pct" in result:
        result["dewpoint_c"] = result["temp_c"] - (100.0 - result["rh_pct"]) * 0.1
    else:
        result["dewpoint_c"] = pd.NA
    return result


def add_precip_periods(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decumulate accumulated precipitation fields into incremental periods.
    """

    result = df.copy()
    if "qpf_in_accum" in result:
        result["qpf_in"] = result["qpf_in_accum"].diff().fillna(0.0)
    else:
        result["qpf_in"] = pd.NA
    return result


def add_wetbulb(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate wet-bulb temperature using a simplified approximation.
    """

    result = df.copy()
    if "temp_c" in result and "rh_pct" in result:
        result["wetbulb_c"] = result["temp_c"] - (100.0 - result["rh_pct"]) * 0.05
    else:
        result["wetbulb_c"] = pd.NA
    return result
