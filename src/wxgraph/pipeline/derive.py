"""Derived field calculators for point forecasts."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from wxgraph.config import get_snow_ratio_method


def add_dewpoint(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute dewpoint (°C) from temperature and humidity.
    """

    result = df.copy()
    if "temp_c" not in result or "rh_percent" not in result:
        result["dewpoint_c"] = pd.NA
        return result

    dew = _dewpoint_from_temp_rh(result["temp_c"].to_numpy(dtype=float), result["rh_percent"].to_numpy(dtype=float))
    result["dewpoint_c"] = dew
    return result


def add_precip_periods(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive incremental precipitation from accumulated totals and classify the type.
    """

    result = df.copy()
    if "valid_time" in result:
        result = result.sort_values("valid_time").reset_index(drop=True)

    accum = result.get("qpf_in_raw") if "qpf_in_raw" in result else result.get("qpf_in")
    if accum is None:
        result["qpf_in"] = 0.0
    else:
        series = pd.Series(accum, dtype=float)
        increments = series.diff().fillna(series.iloc[0] if not series.empty else 0.0).clip(lower=0.0)
        result["qpf_in"] = increments

    temp = result.get("temp_c")
    qpf = result["qpf_in"].to_numpy(dtype=float)
    if temp is not None:
        temp_c = np.asarray(temp, dtype=float)
        precip_type = _classify_precip_type(temp_c, qpf)
        result["precip_type"] = precip_type
        snowfall_in, snow_ratio = _snowfall_from_qpf(temp_c, qpf, precip_type)
        result["snow_ratio"] = snow_ratio
        result["snowfall_in"] = snowfall_in
        result["snow_acc_in"] = result["snowfall_in"].cumsum()
    else:
        result["precip_type"] = "unknown"
        result["snow_ratio"] = 0.0
        result["snowfall_in"] = 0.0
        result["snow_acc_in"] = 0.0

    return result


def add_wetbulb(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate wet-bulb temperature (°C) using the Stull approximation.
    """

    result = df.copy()
    if "temp_c" not in result or "rh_percent" not in result:
        result["wetbulb_c"] = pd.NA
        return result

    result["wetbulb_c"] = _stull_wetbulb(
        result["temp_c"].to_numpy(dtype=float),
        result["rh_percent"].to_numpy(dtype=float),
    )
    return result


def _dewpoint_from_temp_rh(temp_c: Iterable[float], rh_pct: Iterable[float]) -> np.ndarray:
    temp = np.asarray(temp_c, dtype=float)
    rh = np.asarray(rh_pct, dtype=float)
    alpha = np.log(np.clip(rh, 0.1, 100.0) / 100.0) + (17.27 * temp) / (temp + 237.3)
    return (237.3 * alpha) / (17.27 - alpha)


def _stull_wetbulb(temp_c: Iterable[float], rh_pct: Iterable[float]) -> np.ndarray:
    temp = np.asarray(temp_c, dtype=float)
    rh = np.clip(np.asarray(rh_pct, dtype=float), 1.0, 100.0)
    wb = (
        temp * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(temp + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * np.power(rh, 1.5) * np.arctan(0.023101 * rh)
        - 4.686035
    )
    return wb


def _classify_precip_type(temp_c: np.ndarray, qpf_in: np.ndarray) -> np.ndarray:
    classification = np.full_like(temp_c, "rain", dtype=object)
    snow_mask = (temp_c <= 0.5) & (qpf_in > 0)
    mix_mask = (temp_c > 0.5) & (temp_c <= 3.0) & (qpf_in > 0)
    classification[snow_mask] = "snow"
    classification[mix_mask] = "mix"
    classification[qpf_in == 0] = "none"
    return classification


def _snowfall_from_qpf(
    temp_c: np.ndarray,
    qpf_in: np.ndarray,
    precip_type: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    method = get_snow_ratio_method()
    temp_f = temp_c * 9.0 / 5.0 + 32.0
    snow_mask = precip_type == "snow"
    if method == "baxter":
        slr = 10.0 + ((32.0 - temp_f) ** 2) / 50.0
        slr = np.clip(slr, 1.0, 50.0)
        slr = np.where(snow_mask, slr, 0.0)
    else:
        slr = np.where(snow_mask, 10.0, 0.0)
    snowfall_in = qpf_in * slr
    return snowfall_in, slr
