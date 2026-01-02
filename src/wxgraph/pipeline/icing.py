"""Icing diagnostics and LCR computations."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

MM_TO_IN = 0.0393701
MS_TO_MPH = 2.237
FREEZING_CUTOFF_IN = 0.003


def _kelvin_to_c(values: np.ndarray) -> np.ndarray:
    return values - 273.15


def _c_to_f(values: np.ndarray) -> np.ndarray:
    return values * 9.0 / 5.0 + 32.0


def _dewpoint_from_temp_rh(temp_c: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
    result = np.full_like(temp_c, np.nan, dtype=float)
    mask = (~np.isnan(temp_c)) & (~np.isnan(rh_pct)) & (rh_pct > 0)
    if not np.any(mask):
        return result
    t = temp_c[mask]
    rh = rh_pct[mask]
    alpha = np.log(rh / 100.0) + (17.27 * t) / (237.7 + t)
    result[mask] =  (237.7 * alpha) / (17.27 - alpha)
    return result


def _stull_wet_bulb(temp_c: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
    result = np.full_like(temp_c, np.nan, dtype=float)
    mask = (~np.isnan(temp_c)) & (~np.isnan(rh_pct))
    if not np.any(mask):
        return result
    t = temp_c[mask]
    rh = rh_pct[mask]
    result[mask] = (
        t * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(t + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * np.power(rh, 1.5) * np.arctan(0.023101 * rh)
        - 4.686035
    )
    return result


def _rolling_history(values: pd.Series, dt_hours: float) -> np.ndarray:
    if values.empty:
        return values.to_numpy()
    if dt_hours <= 0:
        window = 1
    else:
        window = max(1, int(round(6.0 / dt_hours)))
    return values.rolling(window=window, min_periods=1).mean().to_numpy()


def _decumulate(accum: np.ndarray) -> np.ndarray:
    if len(accum) == 0:
        return accum
    diffs = np.diff(accum, prepend=accum[0])
    return np.maximum(diffs, 0.0)


def _per_hour(values: np.ndarray, dt_hours: float) -> np.ndarray:
    if dt_hours <= 0:
        dt_hours = 1.0
    return values / dt_hours


def _apparent_temp_f(temp_f: np.ndarray, wind_mph: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
    apparent = temp_f.copy()
    wind = np.maximum(wind_mph, 0.0)
    rh = np.clip(rh_pct, 0.0, 100.0)

    wind_chill_mask = (temp_f <= 50.0) & (wind >= 3.0)
    if np.any(wind_chill_mask):
        v16 = np.power(wind[wind_chill_mask], 0.16)
        t = temp_f[wind_chill_mask]
        apparent[wind_chill_mask] = 35.74 + 0.6215 * t - 35.75 * v16 + 0.4275 * t * v16

    heat_mask = (temp_f >= 80.0) & (rh >= 40.0)
    if np.any(heat_mask):
        t = temp_f[heat_mask]
        r = rh[heat_mask]
        hi = (
            -42.379
            + 2.04901523 * t
            + 10.14333127 * r
            - 0.22475541 * t * r
            - 0.00683783 * t * t
            - 0.05481717 * r * r
            + 0.00122874 * t * t * r
            + 0.00085282 * t * r * r
            - 0.00000199 * t * t * r * r
        )
        apparent[heat_mask] = hi

    return apparent


def add_icing_fields(
    df: pd.DataFrame,
    *,
    prefix: str,
    time_col: str = "valid_time",
    temp_k_col: Optional[str] = None,
    rh_pct_col: Optional[str] = None,
    dp_k_col: Optional[str] = None,
    qpf_accum_mm_col: Optional[str] = None,
    snow_accum_mm_col: Optional[str] = None,
    frzr_flag_col: Optional[str] = None,
    snow_flag_col: Optional[str] = None,
    gust_ms_col: Optional[str] = None,
    gust_mph_col: Optional[str] = None,
    wind_mph_col: Optional[str] = None,
    cloud_pct_col: Optional[str] = None,
    latitude: Optional[float] = None,
) -> pd.DataFrame:
    """Append prefixed icing/LCR columns to df."""

    df = df.copy()
    prefix = prefix.upper().replace(" ", "_").replace("-", "_")

    times = pd.to_datetime(df[time_col])
    df[time_col] = times
    df = df.sort_values(time_col).reset_index(drop=True)
    times = df[time_col]
    diffs = times.diff().dt.total_seconds().dropna()
    if diffs.empty:
        dt_hours = 1.0
    else:
        dt_hours = float(np.median(diffs)) / 3600.0
    if not np.isfinite(dt_hours) or dt_hours <= 0:
        dt_hours = 1.0

    n = len(df)
    def get_col(name: Optional[str]) -> np.ndarray:
        if name and name in df:
            return df[name].to_numpy(dtype=float)
        return np.full(n, np.nan, dtype=float)

    temp_k = get_col(temp_k_col)
    temp_c = temp_k - 273.15
    temp_f = _c_to_f(temp_c)
    rh_pct = get_col(rh_pct_col)
    dp_k = get_col(dp_k_col)
    dp_c = np.where(~np.isnan(dp_k), dp_k - 273.15, np.nan)
    dp_c = np.where(np.isnan(dp_c), _dewpoint_from_temp_rh(temp_c, rh_pct), dp_c)
    wb_c = _stull_wet_bulb(temp_c, rh_pct)
    wb_f = _c_to_f(wb_c)
    temp_hist6h_c = _rolling_history(pd.Series(temp_c), dt_hours)
    temp_hist6h_f = _c_to_f(temp_hist6h_c)

    qpf_accum_mm = get_col(qpf_accum_mm_col)
    qpf_accum_mm = np.where(np.isnan(qpf_accum_mm), 0.0, qpf_accum_mm)
    qpf_in_accum = qpf_accum_mm * MM_TO_IN
    ipf_in = _decumulate(qpf_in_accum)
    ipf_in_per_hr = _per_hour(ipf_in, dt_hours)

    snow_accum_mm = get_col(snow_accum_mm_col)
    snow_accum_mm = np.where(np.isnan(snow_accum_mm), 0.0, snow_accum_mm)
    snow_in_accum = snow_accum_mm * MM_TO_IN
    snowsfc_in = _decumulate(snow_in_accum)
    snowsfc_in_per_hr = _per_hour(snowsfc_in, dt_hours)

    snptype = np.zeros(n)
    if snow_flag_col and snow_flag_col in df:
        snptype = df[snow_flag_col].astype(float).to_numpy()
    else:
        snptype = np.where(snowsfc_in > 0, 1.0, 0.0)

    zrptype = np.zeros(n)
    if frzr_flag_col and frzr_flag_col in df:
        zrptype = df[frzr_flag_col].astype(float).to_numpy()

    gust_mph = get_col(gust_mph_col)
    if np.isnan(gust_mph).all():
        gust_mph = get_col(gust_ms_col) * MS_TO_MPH
    wind_mph = get_col(wind_mph_col)
    if np.isnan(wind_mph).all():
        wind_mph = gust_mph
    cloud_pct = get_col(cloud_pct_col)

    def set_col(name: str, values: np.ndarray):
        df[f"{prefix}_{name}"] = values

    set_col("temp_c", temp_c)
    set_col("temp_f", temp_f)
    set_col("temp_hist6h_c", temp_hist6h_c)
    set_col("temp_hist6h_f", temp_hist6h_f)
    set_col("wind10m_mph", wind_mph)
    set_col("rh_pct", rh_pct)
    set_col("dp_c", dp_c)
    set_col("dp_f", _c_to_f(dp_c))
    set_col("wb_c", wb_c)
    set_col("wb_f", wb_f)
    apparent_f = _apparent_temp_f(temp_f, wind_mph, rh_pct)
    set_col("apparent_temp_f", apparent_f)
    set_col("apparent_temp_c", (apparent_f - 32.0) * 5.0 / 9.0)
    set_col("qpf_in_accum", qpf_in_accum)
    set_col("ipf_in", ipf_in)
    set_col("ipf_in_per_hr", ipf_in_per_hr)
    set_col("snow_in_accum", snow_in_accum)
    set_col("snowsfc_in", snowsfc_in)
    set_col("snowsfc_in_per_hr", snowsfc_in_per_hr)
    set_col("snptype", snptype)
    set_col("zrptype", zrptype)
    set_col("gust_mph", gust_mph)
    set_col("cloud_pct", cloud_pct)
    set_col("dt_hours", np.full(n, dt_hours))

    # CIP/BFP/NFP/AFP
    cip = np.where(temp_f <= 29.9, ipf_in_per_hr, 0.0)
    bfp = np.where((temp_f > 29.9) & (temp_f <= 32.9), ipf_in_per_hr, 0.0)
    nfp = np.where((temp_f > 32.9) & (temp_f <= 38.0), ipf_in_per_hr, 0.0)
    afp = np.where(temp_f > 38.0, ipf_in_per_hr, 0.0)
    for arr in (cip, bfp, nfp, afp):
        np.copyto(arr, np.where(arr < FREEZING_CUTOFF_IN, 0.0, arr))
    set_col("cip_inph", cip)
    set_col("bfp_inph", bfp)
    set_col("nfp_inph", nfp)
    set_col("afp_inph", afp)
    set_col("cip_in", cip * dt_hours)
    set_col("bfp_in", bfp * dt_hours)
    set_col("nfp_in", nfp * dt_hours)
    set_col("afp_in", afp * dt_hours)

    # LCR baseline
    lcr = np.zeros(n)
    lcron = np.zeros(n)

    mask_rain = (ipf_in_per_hr > FREEZING_CUTOFF_IN) & (snowsfc_in_per_hr == 0)
    conds = [
        (mask_rain & (ipf_in_per_hr < 0.05) & (temp_f <= 36.0) & (wb_f <= 34.0)),
        (mask_rain & (ipf_in_per_hr >= 0.05) & (ipf_in_per_hr < 0.1) & (temp_f <= 36.0) & (wb_f <= 34.0)),
        (mask_rain & (ipf_in_per_hr >= 0.1) & (ipf_in_per_hr < 0.25) & (temp_f <= 36.0) & (wb_f <= 34.0)),
        (mask_rain & (ipf_in_per_hr >= 0.25) & (temp_f <= 36.0) & (wb_f <= 34.0)),
    ]
    levels = [1, 2, 3, 4]
    for cond, level in zip(conds, levels):
        lcr = np.where(cond, level, lcr)
        lcron = np.where(cond, 1.0, lcron)

    mask_snow = snowsfc_in_per_hr > 0
    snow_levels = [1, 2, 3, 4]
    snow_ranges = [(0.05, 0.3), (0.3, 0.8), (0.8, 1.5), (1.5, None)]
    for level, (low, high) in zip(snow_levels, snow_ranges):
        cond = mask_snow & (temp_f <= 38.0) & (wb_f <= 36.0)
        if low is not None:
            cond &= snowsfc_in_per_hr >= low
        if high is not None:
            cond &= snowsfc_in_per_hr < high
        cond &= lcr < level
        lcr = np.where(cond, level, lcr)
        lcron = np.where(cond, 1.0, lcron)

    active = lcron == 1
    lcr = np.where(active & ((temp_f <= 32.9) | (zrptype > 0)), lcr + 1, lcr)
    lcr = np.where(active & (temp_hist6h_f < 32.0), lcr + 1, lcr)
    sweet = active & (((temp_f >= 20.0) & (temp_f <= 29.9)) | (zrptype > 0))
    lcr = np.where(sweet, lcr + 2, lcr)
    lcr = np.where(active & (zrptype > 0) & (ipf_in_per_hr > FREEZING_CUTOFF_IN), lcr + 2, lcr)
    if latitude is not None:
        lat_abs = abs(latitude)
        if lat_abs <= 35:
            lcr = np.where(active, lcr + 1, lcr)
        if lat_abs <= 34:
            lcr = np.where(active, lcr + 1, lcr)
        if lat_abs <= 33:
            lcr = np.where(active, lcr + 1, lcr)
    lcr = np.where(active & (gust_mph >= 20.0) & (lcr >= 5), lcr + 1, lcr)
    if latitude is not None:
        snow_cap = (latitude > 35) & (snptype == 1) & (zrptype == 0)
        lcr = np.where(snow_cap & (lcr > 7), 7, lcr)

    # Freezing fog factor when no baseline
    inactive = lcron < 1
    fog_conditions = [
        (inactive & (rh_pct > 90) & (temp_f <= 32), 1),
        (inactive & (rh_pct > 93) & (temp_f <= 30), 2),
        (inactive & (rh_pct > 97) & (temp_f <= 32), 2),
        (inactive & (rh_pct > 96) & (temp_f <= 29), 3),
        (inactive & (rh_pct > 98) & (temp_f <= 30), 3),
        (inactive & (rh_pct > 98) & (temp_f <= 27), 4),
    ]
    for cond, level in fog_conditions:
        lcr = np.where(cond, np.maximum(lcr, level), lcr)

    # Clear-sky frost cap
    lcr = np.where((cloud_pct < 10) & (lcr > 3), 3, lcr)

    set_col("lcr", lcr)
    set_col("lcron", lcron)

    return df
