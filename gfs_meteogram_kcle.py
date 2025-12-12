#!/usr/bin/env python3
"""
Multi-panel meteogram for a single point using GFS 0.25° data.

Features
--------
* Downloads a small subset of GFS using the NOMADS grib-filter API.
* Extracts the nearest grid point to a requested lat/lon.
* Builds a pandas DataFrame with temperature, dewpoint/humidity, QPF,
  snowfall with compaction, wind, and apparent temperature.
* Produces a multi-panel meteogram PNG (and optional CSV) for easy review.

This module is structured around a model class so additional point
forecast sources can be plugged in later.
"""

import argparse
import datetime as dt
import logging
import os
from pathlib import Path
from typing import Iterable, Protocol

import cfgrib
from cfgrib.dataset import DatasetBuildError
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
import pandas as pd
import requests
import xarray as xr

BASE_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
AIGFS_BASE_URLS = [
    "https://nomads.ncep.noaa.gov/pub/data/nccf/com/aigfs/v1.0",
    "https://nomads.ncep.noaa.gov/pub/data/nccf/com/aigfs/para",
]
HGEFS_BASE_URLS = [
    "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hgefs/v1.0",
    "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hgefs/para",
]
NAM_BASE_URL = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/nam/prod"
HRRR_BASE_URL = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod"
RAP_BASE_URL = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/rap/prod"
GEFS_BASE_URL = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gefs/prod"
GFS_PROD_BASE_URL = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod"

LOG_LEVEL = os.environ.get("WXGRAPH_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("cfgrib").setLevel(logging.ERROR)
logging.getLogger("cfgrib.messages").setLevel(logging.ERROR)

# Default KCLE coordinates
DEFAULT_LAT = 41.48
DEFAULT_LON = -81.81

# Small lat/lon box around KCLE for the grib filter (degrees)
LAT_TOP = 43.0
LAT_BOTTOM = 40.0
LON_LEFT = -84.0
LON_RIGHT = -79.0

MODEL_HOUR_LIMITS: dict[str, int] = {
    "nam": 84,
    "hrrr": 48,
    "rap": 39,
    "hgefs": 162,
}


def get_env_float(name: str, fallback: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return fallback
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid float for %s=%s; using %s", name, value, fallback)
        return fallback


def get_env_str(name: str, fallback: str) -> str:
    return os.environ.get(name, fallback)


def get_env_list(name: str, fallback: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if not raw:
        return fallback
    parsed = [item.strip() for item in raw.split(",") if item.strip()]
    return parsed or fallback


class PointModelForecast(Protocol):
    def fetch(self, *, no_cache: bool = False) -> None:
        ...

    def to_dataframe(self, *, include_raw_snow_vars: bool = False) -> pd.DataFrame:
        ...

    @property
    def model_name(self) -> str:
        ...


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def choose_default_cycle(now_utc: dt.datetime) -> int:
    """Choose most recent GFS cycle (0, 6, 12, 18 UTC) at or before current UTC."""

    for c in [18, 12, 6, 0]:
        if now_utc.hour >= c:
            return c
    return 18  # previous day fallback (we'll also adjust date below)


def step_back_cycle(run_date: dt.date, cycle: int) -> tuple[dt.date, int]:
    """Return the previous available model cycle (6-hour steps)."""

    if cycle == 0:
        return run_date - dt.timedelta(days=1), 18
    return run_date, cycle - 6


def build_gfs_url(run_date: dt.date, cycle: int, f_hour: int) -> str:
    """
    Build the NOMADS grib-filter URL for GFS 0.25° with required variables.
    """

    ymd = run_date.strftime("%Y%m%d")
    file_name = f"gfs.t{cycle:02d}z.pgrb2.0p25.f{f_hour:03d}"

    params = {
        "file": file_name,
        # variables
        "lev_2_m_above_ground": "on",
        "lev_10_m_above_ground": "on",
        "lev_surface": "on",
        "var_TMP": "on",
        "var_DPT": "on",
        "var_SPFH": "on",
        "var_UGRD": "on",
        "var_VGRD": "on",
        "var_APCP": "on",
        "var_PRMSL": "on",
        "var_GUST": "on",
        # subregion
        "subregion": "",
        "leftlon": str(LON_LEFT),
        "rightlon": str(LON_RIGHT),
        "toplat": str(LAT_TOP),
        "bottomlat": str(LAT_BOTTOM),
        # directory
        "dir": f"/gfs.{ymd}/{cycle:02d}/atmos",
    }

    return requests.Request("GET", BASE_URL, params=params).prepare().url


def build_gfs_direct_url(run_date: dt.date, cycle: int, f_hour: int) -> str:
    """Direct file URL for GFS 0.25° output on NOMADS."""

    ymd = run_date.strftime("%Y%m%d")
    return (
        f"{GFS_PROD_BASE_URL}/gfs.{ymd}/{cycle:02d}/atmos/"
        f"gfs.t{cycle:02d}z.pgrb2.0p25.f{f_hour:03d}"
    )


def build_aigfs_sfc_url(run_date: dt.date, cycle: int, f_hour: int, base_url: str) -> str:
    """Build the NOMADS URL for AIGFS surface deterministic GRIB."""

    ymd = run_date.strftime("%Y%m%d")
    file_name = f"aigfs.t{cycle:02d}z.sfc.f{f_hour:03d}.grib2"
    return (
        f"{base_url}/"
        f"aigfs.{ymd}/{cycle:02d}/model/atmos/grib2/{file_name}"
    )


def build_hgefs_sfc_avg_url(run_date: dt.date, cycle: int, f_hour: int, base_url: str) -> str:
    """Build the NOMADS URL for HGEFS ensemble-mean surface statistics."""

    ymd = run_date.strftime("%Y%m%d")
    file_name = f"hgefs.t{cycle:02d}z.sfc.avg.f{f_hour:03d}.grib2"
    return (
        f"{base_url}/"
        f"hgefs.{ymd}/{cycle:02d}/ensstat/products/atmos/grib2/{file_name}"
    )


def build_hgefs_sfc_spr_url(run_date: dt.date, cycle: int, f_hour: int, base_url: str) -> str:
    """Build the NOMADS URL for HGEFS ensemble spread surface statistics."""

    ymd = run_date.strftime("%Y%m%d")
    file_name = f"hgefs.t{cycle:02d}z.sfc.spr.f{f_hour:03d}.grib2"
    return (
        f"{base_url}/"
        f"hgefs.{ymd}/{cycle:02d}/ensstat/products/atmos/grib2/{file_name}"
    )


def build_nam_url(run_date: dt.date, cycle: int, f_hour: int) -> str:
    """NAM 12 km deterministic surface file.

    NAM surface files on NOMADS follow the pattern:
      nam.tCCz.awphysHH.tm00.grib2
    where HH is 2-digit forecast hour (00, 03, 06, ...).
    """

    ymd = run_date.strftime("%Y%m%d")
    fh_str = f"{f_hour:02d}"  # 00, 03, 06, ... 84

    return (
        f"{NAM_BASE_URL}/nam.{ymd}/"
        f"nam.t{cycle:02d}z.awphys{fh_str}.tm00.grib2"
    )


def build_hrrr_url(run_date: dt.date, cycle: int, f_hour: int) -> str:
    """HRRR deterministic CONUS surface file."""

    ymd = run_date.strftime("%Y%m%d")
    return (
        f"{HRRR_BASE_URL}/hrrr.{ymd}/conus/"
        f"hrrr.t{cycle:02d}z.wrfsfcf{f_hour:02d}.grib2"
    )


def build_rap_url(run_date: dt.date, cycle: int, f_hour: int) -> str:
    """
    RAP deterministic surface file (13 km AWP130 grid).

    RAP files on NOMADS typically follow:
      rap.YYYYMMDD/rap.tCCz.awp130pgrbfHH.grib2
    where HH is 2-digit forecast hour.
    """

    ymd = run_date.strftime("%Y%m%d")
    fh_str = f"{f_hour:02d}"

    return (
        f"{RAP_BASE_URL}/rap.{ymd}/"
        f"rap.t{cycle:02d}z.awp130pgrbf{fh_str}.grib2"
    )


def build_gefs_control_url(run_date: dt.date, cycle: int, f_hour: int) -> str:
    """GEFS control member surface file (0.5°)."""

    ymd = run_date.strftime("%Y%m%d")
    return (
        f"{GEFS_BASE_URL}/gefs.{ymd}/{cycle:02d}/gec00/pgrb2ap5/"
        f"gec00.t{cycle:02d}z.pgrb2a.0p50.f{f_hour:03d}"
    )


def build_gefs_mean_url(run_date: dt.date, cycle: int, f_hour: int) -> str:
    """GEFS ensemble mean surface statistics file."""

    ymd = run_date.strftime("%Y%m%d")
    return (
        f"{GEFS_BASE_URL}/gefs.{ymd}/{cycle:02d}/ensstat/"
        f"gefs.t{cycle:02d}z.pgrb2b_mean.f{f_hour:03d}"
    )


def download_grib(url: str, out_path: Path, *, no_cache: bool = False) -> None:
    """Download the GRIB file at url to out_path."""

    if out_path.exists() and not no_cache:
        # Check if file was downloaded within the last hour
        file_age = dt.datetime.now() - dt.datetime.fromtimestamp(out_path.stat().st_mtime)
        if file_age < dt.timedelta(hours=1):
            print(f"[download] Using cached {out_path.name} (age: {file_age.total_seconds() / 60:.1f} min)")
            return

    print(f"[download] Fetching {out_path.name}")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)


def _is_missing_http_error(exc: requests.HTTPError) -> bool:
    """Return True if HTTP error indicates resource unavailable (403/404)."""

    if exc.response is None:
        return False
    return exc.response.status_code in {403, 404}


def _is_gfs_run_available(run_date: dt.date, cycle: int) -> bool:
    """Check whether a GFS run exists on NOMADS by HEAD-ing the f000 file."""

    url = build_gfs_direct_url(run_date, cycle, 0)
    try:
        resp = requests.head(url, timeout=10)
    except requests.RequestException as exc:
        logger.debug("GFS availability check failed for %s t%02dz: %s", run_date, cycle, exc)
        return False

    if resp.status_code == 200:
        return True
    if resp.status_code in (403, 404):
        return False
    logger.debug(
        "GFS availability check returned status %s for %s t%02dz",
        resp.status_code,
        run_date,
        cycle,
    )
    return False


def find_latest_available_gfs_run(
    run_date: dt.date, cycle: int, *, max_backtrack: int = 16
) -> tuple[dt.date, int] | None:
    """Return the most recent available GFS run by stepping backward if needed."""

    attempts = 0
    date = run_date
    cyc = cycle
    while attempts <= max_backtrack:
        if _is_gfs_run_available(date, cyc):
            return date, cyc
        date, cyc = step_back_cycle(date, cyc)
        attempts += 1
    return None


def get_coord_names(ds: xr.Dataset):
    """Find the latitude/longitude coordinate names in a cfgrib dataset."""

    if "latitude" in ds.coords:
        lat_name = "latitude"
    elif "lat" in ds.coords:
        lat_name = "lat"
    else:
        raise KeyError("Could not find latitude coordinate in dataset.")

    if "longitude" in ds.coords:
        lon_name = "longitude"
    elif "lon" in ds.coords:
        lon_name = "lon"
    else:
        raise KeyError("Could not find longitude coordinate in dataset.")

    return lat_name, lon_name


def select_var(ds: xr.Dataset, candidates: Iterable[str]):
    """Pick the first variable name in 'candidates' that exists in ds."""

    for name in candidates:
        if name in ds.data_vars:
            return ds[name]
    raise KeyError(f"None of {candidates} found in dataset vars: {list(ds.data_vars)}")


def _looks_cumulative(series: pd.Series) -> bool:
    """Heuristic: cumulative series should generally be non-decreasing."""

    diffs = series.diff().dropna()
    if diffs.empty:
        return True
    negative_fraction = (diffs < -1e-3).sum() / len(diffs)
    return negative_fraction < 0.2


def derive_period_from_series(series: pd.Series, *, allow_negative: bool = False) -> pd.Series:
    """Convert a cumulative series into per-period increments."""

    if series.isna().all():
        return pd.Series(np.nan, index=series.index)

    if not _looks_cumulative(series):
        values = series.copy()
    else:
        values = series.diff()
        values.iloc[0] = series.iloc[0]

    if not allow_negative:
        values = values.clip(lower=0.0)
    return values


def limit_forecast_hours(model_key: str, hours: list[int]) -> list[int]:
    """Apply per-model hour caps and emit a single summary message if truncated."""

    limit = MODEL_HOUR_LIMITS.get(model_key)
    if limit is None:
        return list(hours)

    filtered = [h for h in hours if h <= limit]
    label = model_key.upper()
    if hours and max(hours) > limit:
        print(f"[{label}] Requested max fh={max(hours)} but {label} supports <= {limit}; truncating.")

    if not filtered:
        print(f"[{label}] No requested forecast hours fall within <= {limit}; skipping model.")
        return []
    return filtered


def open_point_dataset(path: Path, lat: float, lon: float, filter_by_keys: dict) -> xr.Dataset | None:
    """
    Open a cfgrib dataset filtered by keys and select the nearest point.
    Handles stale index files by deleting and retrying once.
    """

    backend_kwargs = {"filter_by_keys": filter_by_keys}
    idx_path = Path(f"{path}.idx")

    def _open() -> xr.Dataset:
        return xr.open_dataset(path, engine="cfgrib", backend_kwargs=backend_kwargs)

    try:
        ds = _open()
    except (EOFError, DatasetBuildError) as exc:
        if idx_path.exists():
            logger.warning("Removing stale cfgrib index %s due to error: %s", idx_path, exc)
            idx_path.unlink()
            ds = _open()
        else:
            raise

    if not ds.data_vars:
        ds.close()
        return None

    for coord_name in ("latitude", "lat"):
        if coord_name in ds and coord_name not in ds.coords:
            ds = ds.assign_coords({coord_name: ds[coord_name]})
    for coord_name in ("longitude", "lon"):
        if coord_name in ds and coord_name not in ds.coords:
            ds = ds.assign_coords({coord_name: ds[coord_name]})

    lat_name, lon_name = get_coord_names(ds)

    def _nearest_point(dataset: xr.Dataset) -> xr.Dataset:
        lat_da = dataset[lat_name]
        lon_da = dataset[lon_name]
        lat_vals = np.asarray(lat_da)
        lon_vals = np.asarray(lon_da)

        lon_min = float(np.nanmin(lon_vals))
        lon_max = float(np.nanmax(lon_vals))
        use_360 = lon_min >= 0.0 and lon_max > 180.0
        lon_target = lon + 360.0 if (lon < 0.0 and use_360) else lon

        if lat_vals.ndim == lon_vals.ndim == 0:
            logger.debug(
                "Point select: path=%s lon_range=(%.2f, %.2f) requested=(%.2f, %.2f) normalized_lon=%.2f (scalar grid)",
                path,
                lon_min,
                lon_max,
                lat,
                lon,
                lon_target,
            )
            return dataset.load()

        if lat_vals.shape == lon_vals.shape:
            dist = (lat_vals - lat) ** 2 + (lon_vals - lon_target) ** 2
            flat_idx = int(np.nanargmin(dist))
            unraveled = np.unravel_index(flat_idx, lat_vals.shape)
            indexers: dict[str, int] = {}
            for dim, idx_val in zip(lat_da.dims, unraveled):
                indexers[dim] = int(idx_val)
            point = dataset.isel(indexers).load()
            selected_lat = float(lat_vals[unraveled])
            selected_lon = float(lon_vals[unraveled])
            logger.debug(
                "Point select: path=%s lon_range=(%.2f, %.2f) requested=(%.2f, %.2f) normalized_lon=%.2f selected=(%.2f, %.2f)",
                path,
                lon_min,
                lon_max,
                lat,
                lon,
                lon_target,
                selected_lat,
                selected_lon,
            )
            return point

        if lat_vals.ndim == 1 and lon_vals.ndim == 1:
            idx_lat = int(np.nanargmin(np.abs(lat_vals - lat)))
            idx_lon = int(np.nanargmin(np.abs(lon_vals - lon_target)))
            lat_dim = lat_da.dims[0]
            lon_dim = lon_da.dims[0]
            point = dataset.isel({lat_dim: idx_lat, lon_dim: idx_lon}).load()
            selected_lat = float(lat_vals[idx_lat])
            selected_lon = float(lon_vals[idx_lon])
            logger.debug(
                "Point select: path=%s lon_range=(%.2f, %.2f) requested=(%.2f, %.2f) normalized_lon=%.2f selected=(%.2f, %.2f)",
                path,
                lon_min,
                lon_max,
                lat,
                lon,
                lon_target,
                selected_lat,
                selected_lon,
            )
            return point

        raise ValueError("Latitude/longitude coordinate shapes do not match.")

    try:
        pt = _nearest_point(ds)
    finally:
        ds.close()
    return pt


def c_to_f(temp_c: np.ndarray) -> np.ndarray:
    return temp_c * 9.0 / 5.0 + 32.0


def rh_from_t_td(t_c: np.ndarray, td_c: np.ndarray) -> np.ndarray:
    es = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
    e = 6.112 * np.exp((17.67 * td_c) / (td_c + 243.5))
    return 100.0 * e / es


def dewpoint_from_spfh(temp_c: np.ndarray, spfh: np.ndarray, pressure_hpa: float = 1013.0) -> np.ndarray:
    """Approximate dewpoint (°C) from specific humidity (kg/kg) and pressure."""

    epsilon = 0.622
    w = spfh / (1.0 - spfh)  # mixing ratio
    e = pressure_hpa * w / (w + epsilon)  # vapor pressure in hPa
    ln_ratio = np.log(e / 6.112)
    td_c = (243.5 * ln_ratio) / (17.67 - ln_ratio)
    return td_c


def apparent_temperature_f(temp_f: np.ndarray, wind_mph: np.ndarray, rh_percent: np.ndarray) -> np.ndarray:
    """
    Combine wind chill and heat index into a single "feels like" temperature.
    - When temp <= 50°F and wind >= 3 mph → Wind Chill
    - When temp >= 80°F → Heat Index
    - Otherwise → temp itself
    """

    temp_f = np.asarray(temp_f)
    wind_mph = np.asarray(wind_mph)
    rh_percent = np.asarray(rh_percent)

    # Wind chill
    wind_chill = 35.74 + 0.6215 * temp_f - 35.75 * np.power(wind_mph, 0.16) + 0.4275 * temp_f * np.power(wind_mph, 0.16)
    wind_chill_mask = (temp_f <= 50.0) & (wind_mph >= 3.0)

    # Heat index (Rothfusz)
    hi = (
        -42.379
        + 2.04901523 * temp_f
        + 10.14333127 * rh_percent
        - 0.22475541 * temp_f * rh_percent
        - 6.83783e-3 * temp_f**2
        - 5.481717e-2 * rh_percent**2
        + 1.22874e-3 * temp_f**2 * rh_percent
        + 8.5282e-4 * temp_f * rh_percent**2
        - 1.99e-6 * temp_f**2 * rh_percent**2
    )
    heat_index_mask = temp_f >= 80.0

    feels_like = temp_f.copy()
    feels_like = np.where(wind_chill_mask, wind_chill, feels_like)
    feels_like = np.where(heat_index_mask, hi, feels_like)
    return feels_like


def wet_bulb_f(temp_f: np.ndarray, rh_percent: np.ndarray) -> np.ndarray:
    """
    Approximate wet-bulb temperature (°F) using the Stull (2011) formulation.
    """

    temp_f = np.asarray(temp_f, dtype=float)
    rh_percent = np.asarray(rh_percent, dtype=float)
    wb = np.full_like(temp_f, np.nan, dtype=float)

    valid = (~np.isnan(temp_f)) & (~np.isnan(rh_percent))
    if not np.any(valid):
        return wb

    temp_c = (temp_f[valid] - 32.0) * 5.0 / 9.0
    rh = np.clip(rh_percent[valid], 1.0, 100.0)
    # Stull approximation (2011) expects temperature in Celsius and RH (%)
    tw_c = (
        temp_c * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(temp_c + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh)
        - 4.686035
    )
    wb[valid] = tw_c * 9.0 / 5.0 + 32.0
    return wb


def precip_temp_partitions(temp_f: np.ndarray, qpf_in: np.ndarray) -> dict[str, np.ndarray]:
    """
    Partition liquid precipitation into temperature bins for icing diagnostics.
    """

    temp = np.asarray(temp_f, dtype=float)
    qpf = np.asarray(qpf_in, dtype=float)
    n = qpf.size
    result = {
        "cip_in": np.zeros(n, dtype=float),
        "bfp_in": np.zeros(n, dtype=float),
        "nfp_in": np.zeros(n, dtype=float),
        "afp_in": np.zeros(n, dtype=float),
    }

    valid = (~np.isnan(temp)) & (~np.isnan(qpf)) & (qpf > 0)
    if not np.any(valid):
        return result

    temp_valid = temp[valid]
    qpf_valid = qpf[valid]

    partitions = {
        "cip_in": temp_valid <= 29.0,
        "bfp_in": (temp_valid > 29.0) & (temp_valid <= 32.0),
        "nfp_in": (temp_valid > 32.0) & (temp_valid <= 38.0),
        "afp_in": temp_valid > 38.0,
    }

    for key, mask in partitions.items():
        values = np.zeros_like(qpf_valid)
        values[mask] = qpf_valid[mask]
        # Apply minimum threshold of 0.003"
        values = np.where(values < 0.003, 0.0, values)
        result[key][valid] = values

    return result


def compute_temp_history_6h(temp_f: pd.Series, valid_time: pd.Series) -> pd.Series:
    """Return a rolling 6-hour mean temperature based on the timestep spacing."""

    if temp_f.empty:
        return temp_f.copy()

    diffs = valid_time.diff().dt.total_seconds().dropna() / 3600.0
    if diffs.empty:
        window = 2
    else:
        median_step = float(np.median(diffs))
        if not np.isfinite(median_step) or median_step <= 0:
            window = 2
        else:
            window = max(1, int(round(6.0 / median_step)))
    return temp_f.rolling(window=window, min_periods=1).mean()


def road_icing_risk_score(
    qpf_in: np.ndarray,
    temp_f: np.ndarray,
    wetbulb_f: np.ndarray,
    snowfall_in: np.ndarray | None,
    rh_percent: np.ndarray | None,
    wind_mph: np.ndarray | None,
    *,
    temp_history_f: np.ndarray | None = None,
    freeze_fog_flag: np.ndarray | None = None,
    cip_flag: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a heuristic 0–7 road icing risk score and simple textual reason.
    """

    qpf = np.asarray(qpf_in, dtype=float)
    temp = np.asarray(temp_f, dtype=float)
    wetbulb = np.asarray(wetbulb_f, dtype=float)
    snowfall = None if snowfall_in is None else np.asarray(snowfall_in, dtype=float)
    rh = None if rh_percent is None else np.asarray(rh_percent, dtype=float)
    wind = None if wind_mph is None else np.asarray(wind_mph, dtype=float)
    hist = None if temp_history_f is None else np.asarray(temp_history_f, dtype=float)

    n = qpf.size
    risk = np.zeros(n, dtype=int)
    reason = np.full(n, "no_hazard", dtype=object)

    valid_liq = (
        (qpf > 0.003)
        & (~np.isnan(temp))
        & (~np.isnan(wetbulb))
        & (temp <= 36.0)
        & (wetbulb <= 34.0)
    )
    liq_amount = np.where(valid_liq, qpf, 0.0)
    liq_score = np.zeros(n, dtype=int)
    liq_score = np.where(liq_amount >= 0.25, 4, liq_score)
    liq_score = np.where((liq_amount >= 0.10) & (liq_amount < 0.25), 3, liq_score)
    liq_score = np.where((liq_amount >= 0.05) & (liq_amount < 0.10), 2, liq_score)
    liq_score = np.where((liq_amount >= 0.003) & (liq_amount < 0.05), 1, liq_score)
    mask = liq_score > 0
    risk = np.where(mask & (liq_score > risk), liq_score, risk)
    reason = np.where(mask & (liq_score >= risk), "liq_precip_near_freezing", reason)

    if snowfall is not None:
        valid_snow = (
            (snowfall > 0)
            & (~np.isnan(temp))
            & (~np.isnan(wetbulb))
            & (temp <= 38.0)
            & (wetbulb <= 36.0)
        )
        snow_amount = np.where(valid_snow, snowfall, 0.0)
        snow_score = np.zeros(n, dtype=int)
        snow_score = np.where(snow_amount >= 1.5, 4, snow_score)
        snow_score = np.where((snow_amount >= 0.8) & (snow_amount < 1.5), 3, snow_score)
        snow_score = np.where((snow_amount >= 0.3) & (snow_amount < 0.8), 2, snow_score)
        snow_score = np.where((snow_amount >= 0.05) & (snow_amount < 0.3), 1, snow_score)
        mask = snow_score > risk
        risk = np.where(mask, snow_score, risk)
        reason = np.where(mask, "snow_hazard", reason)

    cip = (
        cip_flag
        if cip_flag is not None
        else ((qpf > 0.003) & (~np.isnan(temp)) & (temp <= 29.0))
    )
    active = risk > 0
    cold_mask = (~np.isnan(temp)) & (temp <= 32.9)
    increment_mask = active & (cold_mask | cip)
    risk = np.where(increment_mask, risk + 1, risk)

    if hist is not None:
        hist_mask = active & (~np.isnan(hist)) & (hist <= 32.0)
        risk = np.where(hist_mask, risk + 1, risk)

    sweet_spot = active & (~np.isnan(temp)) & (temp >= 20.0) & (temp <= 29.9)
    risk = np.where(sweet_spot, risk + 2, risk)

    if wind is not None:
        windy = (risk >= 5) & (~np.isnan(wind)) & (wind >= 20.0)
        risk = np.where(windy, risk + 1, risk)

    if freeze_fog_flag is None:
        if rh is not None:
            freeze_fog_flag = (
                (np.nan_to_num(qpf) == 0.0)
                & (~np.isnan(rh))
                & (rh >= 97.0)
                & (
                    (~np.isnan(temp) & (temp <= 32.0))
                    | (~np.isnan(wetbulb) & (wetbulb <= 32.0))
                )
            )
        else:
            freeze_fog_flag = np.zeros(n, dtype=bool)

    if rh is not None:
        ff_mask = freeze_fog_flag & (risk == 0)
        ff_risk = np.zeros(n, dtype=int)
        ff_risk = np.where(ff_mask & (rh > 98.0) & (~np.isnan(temp)) & (temp <= 30.0), 3, ff_risk)
        ff_risk = np.where(ff_mask & (rh >= 97.0) & (~np.isnan(temp)) & (temp <= 32.0), np.maximum(ff_risk, 2), ff_risk)
        ff_risk = np.where(ff_mask & (rh > 93.0) & (~np.isnan(temp)) & (temp <= 30.0), np.maximum(ff_risk, 2), ff_risk)
        ff_apply = ff_risk > risk
        risk = np.where(ff_apply, ff_risk, risk)
        reason = np.where(ff_apply & (ff_risk > 0), "freezing_fog_possible", reason)

    risk = np.clip(risk, 0, 7)
    reason = np.where(risk == 0, "no_hazard", reason)
    return risk, reason


def compute_snowfall_and_compaction(
    qpf_in: np.ndarray,
    temp_c: np.ndarray,
    precip_type: np.ndarray,
    *,
    snowfall_native: np.ndarray | None = None,
    snow_depth_native: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute snowfall per period and compacted accumulation.
    Native snowfall/depth data is preferred when available; otherwise it is estimated.
    """

    qpf_in = np.asarray(qpf_in, dtype=float)
    temp_c = np.asarray(temp_c, dtype=float)
    precip_type = np.asarray(precip_type, dtype=object)

    snowfall_from_native = None
    if snowfall_native is not None:
        native_series = pd.Series(snowfall_native)
        if native_series.notna().any():
            snowfall_from_native = derive_period_from_series(native_series, allow_negative=True).to_numpy()

    if snowfall_from_native is not None:
        snowfall = snowfall_from_native
    else:
        snowfall = np.zeros_like(qpf_in)
        cold_mask = temp_c <= -1.0
        cool_mask = (temp_c > -1.0) & (temp_c <= 1.0)
        mix_mask = precip_type == "mix"
        rain_mask = precip_type == "rain"
        ratio = np.zeros_like(qpf_in)
        ratio[cold_mask] = 10.0
        ratio[cool_mask] = np.maximum(ratio[cool_mask], 5.0)
        ratio[mix_mask] = 5.0
        ratio[rain_mask] = 0.0
        snowfall = qpf_in * ratio

    if snow_depth_native is not None:
        depth_series = pd.Series(snow_depth_native).to_numpy()
        if np.all(np.isnan(depth_series)):
            snow_acc = np.cumsum(np.nan_to_num(snowfall))
        else:
            snow_acc = depth_series
    else:
        snow_acc = np.zeros_like(qpf_in)
        depth = 0.0
        for i, (snow_amt, t) in enumerate(zip(snowfall, temp_c)):
            if np.isnan(snow_amt):
                snow_amt = 0.0
            if t > 0.0:
                depth *= 0.92
            else:
                depth *= 0.98
            depth += snow_amt
            snow_acc[i] = depth

    return snowfall, snow_acc


def compare_models(model_dfs: dict[str, pd.DataFrame]) -> None:
    """
    Print basic diagnostics comparing each model to GFS over overlapping hours.
    """

    base_name = next((name for name in model_dfs if name.lower().startswith("gfs")), None)
    if base_name is None:
        logger.info("Skipping diagnostics; GFS baseline not available.")
        return

    base_df = model_dfs[base_name]
    base_subset = base_df[
        ["valid_time", "temp_f", "qpf_in", "wind10m_mph"]
    ].rename(
        columns={
            "temp_f": "temp_f_base",
            "qpf_in": "qpf_in_base",
            "wind10m_mph": "wind_mph_base",
        }
    )

    for name, df in model_dfs.items():
        if name == base_name:
            continue
        other_subset = df[
            ["valid_time", "temp_f", "qpf_in", "wind10m_mph"]
        ].rename(
            columns={
                "temp_f": "temp_f_other",
                "qpf_in": "qpf_in_other",
                "wind10m_mph": "wind_mph_other",
            }
        )
        merged = pd.merge(base_subset, other_subset, on="valid_time", how="inner")
        if merged.empty:
            logger.info("[compare] %s has no overlapping times with %s.", name, base_name)
            continue

        temp_bias = (merged["temp_f_other"] - merged["temp_f_base"]).mean()
        qpf_corr = np.nan
        if merged["qpf_in_other"].notna().sum() >= 2 and merged["qpf_in_base"].notna().sum() >= 2:
            qpf_corr = np.corrcoef(
                merged["qpf_in_base"].fillna(0),
                merged["qpf_in_other"].fillna(0),
            )[0, 1]
        wind_mae = np.abs(merged["wind_mph_other"] - merged["wind_mph_base"]).mean()

        logger.info(
            "[compare] %s vs %s – Temp bias: %.2f°F, QPF corr: %s, Wind MAE: %.2f mph",
            name,
            base_name,
            temp_bias,
            f"{qpf_corr:.2f}" if not np.isnan(qpf_corr) else "n/a",
            wind_mae,
        )


MERGED_FIELDS = [
    "temp_f",
    "apparent_temp_f",
    "qpf_in",
    "snowfall_in",
    "snow_acc_in",
    "wind10m_mph",
    "wetbulb_f",
    "cip_in",
    "bfp_in",
    "nfp_in",
    "afp_in",
    "road_icing_risk",
    "road_icing_reason",
    "freeze_fog_flag",
    "freezing_precip_flag",
    "temp_hist_6h_f",
]


def build_merged_wide_dataframe(
    model_dfs: dict[str, pd.DataFrame], fields: Iterable[str] = MERGED_FIELDS
) -> pd.DataFrame | None:
    """Return a merged wide dataframe with one column per model/field."""

    merged_df: pd.DataFrame | None = None
    for name, df in model_dfs.items():
        available_fields = [field for field in fields if field in df.columns]
        if not available_fields:
            continue
        col_prefix = name.upper().replace("-", "_").replace(" ", "_")
        rename_map = {field: f"{col_prefix}_{field}" for field in available_fields}
        subset = df[["valid_time", *available_fields]].rename(columns=rename_map)
        if merged_df is None:
            merged_df = subset
        else:
            merged_df = merged_df.merge(subset, on="valid_time", how="outer")

    if merged_df is None:
        logger.warning("Unable to create merged dataframe; requested fields missing.")
        return None

    return merged_df.sort_values("valid_time")


def merge_model_csvs(model_dfs: dict[str, pd.DataFrame], out_path: Path) -> None:
    merged_df = build_merged_wide_dataframe(model_dfs)
    if merged_df is None:
        return
    merged_df.to_csv(out_path, index=False)
    print(f"[csv] Saved merged data to {out_path}")


def write_merged_json(model_dfs: dict[str, pd.DataFrame], out_path: Path) -> None:
    merged_df = build_merged_wide_dataframe(model_dfs)
    if merged_df is None:
        return
    merged_df.to_json(out_path, orient="records", date_format="iso")
    print(f"[json] Saved merged data to {out_path}")
def classify_precip_type(temp_c: np.ndarray, qpf_in: np.ndarray) -> np.ndarray:
    precip = []
    for t, q in zip(temp_c, qpf_in):
        if q == 0:
            precip.append("none")
        elif t <= 0.5:
            precip.append("snow")
        elif t <= 2.0:
            precip.append("mix")
        else:
            precip.append("rain")
    return np.array(precip, dtype=object)


def validate_df(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    Ensure required columns exist, fill missing with NaN, and warn on inconsistencies.
    """

    df = df.copy()
    required = [
        "valid_time",
        "temp_c",
        "temp_f",
        "qpf_in",
        "qpf_in_raw",
        "rh_percent",
        "wind10m_mph",
        "wind10m_ms",
        "precip_type",
        "snowfall_in",
        "snow_acc_in",
    ]
    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    df = df.sort_values("valid_time").reset_index(drop=True)

    temps = df["temp_c"]
    if temps.notna().any():
        extreme_mask = (temps < -60.0) | (temps > 60.0)
        if extreme_mask.any():
            logger.warning("[%s] Extreme temperature values detected", model_name)
        median_temp = temps.median()
        if median_temp > 120.0:
            logger.warning("[%s] Temperatures appear to be in Kelvin; converting to Celsius.", model_name)
            df["temp_c"] = df["temp_c"] - 273.15
            if "tdew_c" in df.columns:
                df["tdew_c"] = df["tdew_c"] - 273.15
            df["temp_f"] = c_to_f(df["temp_c"].to_numpy())

    if "tdew_c" in df.columns and "spfh_2m" in df.columns:
        mask = df["tdew_c"].isna() & df["spfh_2m"].notna() & df["temp_c"].notna()
        if mask.any():
            td = dewpoint_from_spfh(df.loc[mask, "temp_c"].to_numpy(), df.loc[mask, "spfh_2m"].to_numpy())
            df.loc[mask, "tdew_c"] = td
            rh_vals = rh_from_t_td(df.loc[mask, "temp_c"].to_numpy(), td)
            df.loc[mask, "rh_percent"] = rh_vals

    if "wind10m_ms" in df.columns and df["wind10m_ms"].notna().any():
        df.loc[df["wind10m_mph"].isna(), "wind10m_mph"] = df["wind10m_ms"] * 2.23694
    elif {"u10m_ms", "v10m_ms"}.issubset(df.columns):
        wind = np.hypot(df["u10m_ms"], df["v10m_ms"])
        df["wind10m_ms"] = wind
        df["wind10m_mph"] = wind * 2.23694

    if "valid_time" in df.columns and df["valid_time"].notna().any():
        diffs = df["valid_time"].diff().dropna()
        if not diffs.empty:
            if (diffs <= dt.timedelta(0)).any():
                logger.warning("[%s] Non-monotonic valid_time values detected.", model_name)
            median_diff = diffs.median()
            if (diffs > median_diff * 2).any():
                logger.warning("[%s] Large gaps found in valid_time series.", model_name)

    if "qpf_in_raw" in df.columns:
        qpf_series = df["qpf_in_raw"].astype(float)
        if (qpf_series < -1e-3).any():
            logger.warning("[%s] Negative QPF values detected; clamping to zero.", model_name)
            qpf_series = qpf_series.clip(lower=0.0)
        df["qpf_in"] = derive_period_from_series(qpf_series)
    elif "qpf_in" in df.columns:
        df["qpf_in"] = df["qpf_in"].clip(lower=0.0)

    if {"temp_c", "qpf_in"}.issubset(df.columns):
        precip_type = classify_precip_type(df["temp_c"].to_numpy(), df["qpf_in"].to_numpy())
        df["precip_type"] = precip_type
        snowfall_native = df["snowfall_native_in_raw"].values if "snowfall_native_in_raw" in df.columns else None
        snow_depth_native = df["snow_depth_native_in"].values if "snow_depth_native_in" in df.columns else None
        snowfall, snow_acc = compute_snowfall_and_compaction(
            df["qpf_in"].to_numpy(),
            df["temp_c"].to_numpy(),
            precip_type,
            snowfall_native=snowfall_native,
            snow_depth_native=snow_depth_native,
        )
        df["snowfall_in"] = snowfall
        df["snow_acc_in"] = snow_acc

    df["qpf_in"] = df["qpf_in"].clip(lower=0.0)
    return df


def extract_surface_record(path: Path, lat: float, lon: float, fh: int, *, gust_var_candidates: Iterable[str] | None = None) -> dict:
    """
    Extract surface-level fields from a GRIB file for the nearest grid point.
    Some products (e.g., NAM) include both instant and accumulated stepTypes
    at the surface, so handle them separately to avoid cfgrib conflicts.
    """

    gust_candidates = tuple(gust_var_candidates) if gust_var_candidates else ("GUST_surface", "GUST")

    pt_surface_inst: xr.Dataset | None = None
    try:
        pt_surface_inst = open_point_dataset(
            path,
            lat,
            lon,
            {"typeOfLevel": "surface", "stepType": "instant"},
        )
    except DatasetBuildError as exc:
        logger.warning(
            "Falling back to generic surface open for instant fields on %s: %s",
            path,
            exc,
        )
        pt_surface_inst = open_point_dataset(path, lat, lon, {"typeOfLevel": "surface"})

    pt_surface_acc: xr.Dataset | None = None
    try:
        pt_surface_acc = open_point_dataset(
            path,
            lat,
            lon,
            {"typeOfLevel": "surface", "stepType": "accum"},
        )
    except DatasetBuildError as exc:
        logger.warning("No accum surface stepType for %s (APCP fallback to instant): %s", path, exc)

    pt_2m = open_point_dataset(path, lat, lon, {"typeOfLevel": "heightAboveGround", "level": 2})
    pt_10m = open_point_dataset(path, lat, lon, {"typeOfLevel": "heightAboveGround", "level": 10})
    pt_mean_sea = open_point_dataset(path, lat, lon, {"typeOfLevel": "meanSea"})

    def select_from_points(points: list[xr.Dataset | None], candidates: Iterable[str]):
        for pt in points:
            if pt is None:
                continue
            for name in candidates:
                if name in pt.data_vars:
                    return pt[name]
        raise KeyError

    # Temperature from 2 m dataset if available
    try:
        t = select_from_points(
            [pt_2m, pt_surface_inst],
            ["t2m", "2t", "TMP_2maboveground", "TMP_surface", "TMP"],
        )
        temp_k = float(t.values)
    except KeyError:
        temp_k = np.nan

    temp_c = temp_k - 273.15 if not np.isnan(temp_k) else np.nan

    rh = np.nan

    dew_c = np.nan
    try:
        dew = select_from_points(
            [pt_2m, pt_surface_inst],
            ["d2m", "2d", "DPT_2maboveground", "DPT_surface", "DPT", "DEPR"],
        )
        dew_val = float(dew.values)
        dew_c = dew_val - 273.15 if dew_val > 150.0 else dew_val
    except KeyError:
        pass

    spfh_val = np.nan
    try:
        spfh_val = float(select_from_points([pt_2m, pt_surface_inst], ["q2m", "SPFH_2maboveground", "SPFH_surface", "SPFH"]).values)
        if np.isnan(dew_c) and not np.isnan(temp_c):
            dew_c = float(dewpoint_from_spfh(np.array([temp_c]), np.array([spfh_val]))[0])
    except (KeyError, ValueError):
        spfh_val = np.nan

    if not np.isnan(temp_c) and not np.isnan(dew_c):
        rh = rh_from_t_td(np.array([temp_c]), np.array([dew_c]))[0]

    # Wind components
    try:
        u = float(select_from_points([pt_surface_inst, pt_10m], ["UGRD_surface", "UGRD", "u10", "10u"]).values)
        v = float(select_from_points([pt_surface_inst, pt_10m], ["VGRD_surface", "VGRD", "v10", "10v"]).values)
    except KeyError:
        u = np.nan
        v = np.nan

    wind_ms = float(np.hypot(u, v)) if not (np.isnan(u) or np.isnan(v)) else np.nan
    wind_mph = wind_ms * 2.23694 if not np.isnan(wind_ms) else np.nan

    gust_mph = np.nan
    try:
        gust = float(select_from_points([pt_surface_inst, pt_10m], gust_candidates).values)
        gust_mph = gust * 2.23694
    except KeyError:
        pass

    try:
        apcp = select_from_points(
            [pt_surface_acc, pt_surface_inst],
            ["APCP_surface", "APCP", "apcp", "tp", "PRATE_surface"],
        )
        qpf_mm = float(apcp.values)
        qpf_in_raw = qpf_mm / 25.4
    except KeyError:
        qpf_in_raw = 0.0

    try:
        prmsl = select_from_points([pt_surface_inst, pt_mean_sea], ["PRMSL", "PRMSL_meansealevel", "prmsl"])
        pr_val = float(prmsl.values)
        mslp_hpa = pr_val / 100.0 if pr_val > 2000.0 else pr_val
    except KeyError:
        mslp_hpa = np.nan

    snowfall_native_in = np.nan
    snow_depth_native_in = np.nan
    try:
        snow = select_from_points(
            [pt_surface_acc, pt_surface_inst],
            ["ASNOW_surface", "ASNOW", "WEASD_surface", "WEASD", "SNOW_surface"],
        )
        snow_val = float(snow.values)
        snowfall_native_in = snow_val / 25.4
    except KeyError:
        pass

    try:
        snowd = select_from_points([pt_surface_inst, pt_surface_acc], ["SNOD_surface", "SNOD", "snowh"])
        snow_depth_native_in = float(snowd.values) * 39.3701
    except KeyError:
        pass

    point_ref = pt_surface_inst or pt_surface_acc or pt_2m or pt_10m or pt_mean_sea
    if point_ref is None:
        raise FileNotFoundError(f"Unable to read any fields from {path}")

    time = point_ref["time"]
    if "step" in point_ref.coords:
        valid_time = (
            pd.to_datetime(time.values)
            + pd.to_timedelta(point_ref["step"].values)
        ).to_pydatetime()
    else:
        valid_time = (
            pd.to_datetime(time.values)
            + pd.to_timedelta(fh, "h")
        ).to_pydatetime()

    return {
        "valid_time": pd.to_datetime(valid_time),
        "f_hour": fh,
        "temp_c": temp_c,
        "temp_f": c_to_f(np.array([temp_c]))[0] if not np.isnan(temp_c) else np.nan,
        "tdew_c": dew_c,
        "rh_percent": rh,
        "qpf_in_raw": qpf_in_raw,
        "snowfall_native_in_raw": snowfall_native_in,
        "snow_depth_native_in": snow_depth_native_in,
        "wind10m_ms": wind_ms,
        "wind10m_mph": wind_mph,
        "u10m_ms": u,
        "v10m_ms": v,
        "gust_mph": gust_mph,
        "mslp_hpa": mslp_hpa,
        "spfh_2m": spfh_val,
    }


def extract_hgefs_spread_record(path: Path, lat: float, lon: float) -> dict:
    """Extract optional spread fields from the HGEFS spread GRIB file."""

    pt_surface = open_point_dataset(path, lat, lon, {"typeOfLevel": "surface"})
    pt_2m = open_point_dataset(path, lat, lon, {"typeOfLevel": "heightAboveGround", "level": 2})
    pt_10m = open_point_dataset(path, lat, lon, {"typeOfLevel": "heightAboveGround", "level": 10})

    def select_from_points(points: list[xr.Dataset | None], candidates: Iterable[str]):
        for pt in points:
            if pt is None:
                continue
            for name in candidates:
                if name in pt.data_vars:
                    return pt[name]
        raise KeyError

    spread_values: dict[str, float] = {}

    try:
        t = select_from_points([pt_2m, pt_surface], ["t2m", "2t", "TMP_2maboveground", "TMP_surface", "TMP"])
        temp_spread_c = float(t.values) - 273.15
        spread_values["temp_spread_c"] = temp_spread_c
        spread_values["temp_spread_f"] = c_to_f(np.array([temp_spread_c]))[0]
    except KeyError:
        pass

    try:
        apcp = select_from_points([pt_surface], ["APCP_surface", "APCP", "apcp", "tp"])
        qpf_mm = float(apcp.values)
        spread_values["qpf_in_spread"] = qpf_mm / 25.4
    except KeyError:
        pass

    try:
        u = float(select_from_points([pt_surface, pt_10m], ["UGRD_surface", "UGRD", "u10", "10u"]).values)
        v = float(select_from_points([pt_surface, pt_10m], ["VGRD_surface", "VGRD", "v10", "10v"]).values)
        wind_spread_ms = float(np.hypot(u, v))
        spread_values["wind10m_spread_ms"] = wind_spread_ms
        spread_values["wind10m_spread_mph"] = wind_spread_ms * 2.23694
    except KeyError:
        pass

    return spread_values


def finalize_point_dataframe(records: list[dict], *, include_raw_snow_vars: bool = False) -> pd.DataFrame:
    """Finalize the list of per-hour records into the canonical dataframe."""

    if not records:
        raise ValueError("No records provided to finalize into a dataframe.")

    df = pd.DataFrame.from_records(records).sort_values("valid_time").reset_index(drop=True)

    if "qpf_in_raw" not in df.columns:
        df["qpf_in_raw"] = 0.0

    df["qpf_in"] = derive_period_from_series(df["qpf_in_raw"])

    precip_type = classify_precip_type(df["temp_c"].values, df["qpf_in"].values)
    df["precip_type"] = precip_type

    snowfall_native = df["snowfall_native_in_raw"].values if "snowfall_native_in_raw" in df.columns else None
    snow_depth_native = df["snow_depth_native_in"].values if "snow_depth_native_in" in df.columns else None

    snowfall, snow_acc = compute_snowfall_and_compaction(
        df["qpf_in"].values,
        df["temp_c"].values,
        precip_type,
        snowfall_native=snowfall_native,
        snow_depth_native=snow_depth_native,
    )
    df["snowfall_in"] = snowfall
    df["snow_acc_in"] = snow_acc

    df["apparent_temp_f"] = apparent_temperature_f(
        df["temp_f"].values,
        df["wind10m_mph"].fillna(0).values,
        df["rh_percent"].fillna(0).values,
    )

    if not include_raw_snow_vars:
        df = df.drop(columns=["snowfall_native_in_raw", "snow_depth_native_in"], errors="ignore")

    return df


def add_road_icing_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """Augment the dataframe with road icing diagnostics."""

    df = df.copy()
    temp_f = df["temp_f"].to_numpy()
    rh = df["rh_percent"].to_numpy() if "rh_percent" in df.columns else np.full(len(df), np.nan)
    qpf = df["qpf_in"].to_numpy()
    snowfall = df["snowfall_in"].to_numpy() if "snowfall_in" in df.columns else None
    wind = df["wind10m_mph"].to_numpy() if "wind10m_mph" in df.columns else None

    df["wetbulb_f"] = wet_bulb_f(temp_f, rh)

    partitions = precip_temp_partitions(temp_f, qpf)
    for name, values in partitions.items():
        df[name] = values

    df["cip_flag"] = df["cip_in"] > 0
    df["bfp_flag"] = df["bfp_in"] > 0

    wetbulb = df["wetbulb_f"].to_numpy()
    rh_series = df["rh_percent"]
    qpf_series = df["qpf_in"]

    freeze_fog_mask = (
        (qpf_series.fillna(0) == 0)
        & (rh_series >= 97.0)
        & (
            (df["temp_f"] <= 32.0)
            | (df["wetbulb_f"] <= 32.0)
        )
    )
    df["freeze_fog_flag"] = freeze_fog_mask.fillna(False)

    df["freezing_precip_flag"] = (
        (qpf_series > 0)
        & (
            (df["temp_f"] <= 32.0)
            | (df["wetbulb_f"] <= 32.0)
        )
    ).fillna(False)

    df["temp_hist_6h_f"] = compute_temp_history_6h(df["temp_f"], df["valid_time"])

    risk, reason = road_icing_risk_score(
        qpf,
        temp_f,
        wetbulb,
        snowfall,
        rh,
        wind,
        temp_history_f=df["temp_hist_6h_f"].to_numpy(),
        freeze_fog_flag=df["freeze_fog_flag"].to_numpy(),
        cip_flag=df["cip_flag"].to_numpy(),
    )
    df["road_icing_risk"] = risk
    df["road_icing_reason"] = reason

    return df


def log_road_icing_summary(model_name: str, df: pd.DataFrame) -> None:
    """Emit a concise summary of the icing risk timeline for sanity checks."""

    if "road_icing_risk" not in df.columns:
        return
    risk = df["road_icing_risk"]
    valid_time = df["valid_time"]
    high_count = int((risk >= 4).sum())
    risk_mask = risk >= 1
    if risk_mask.any():
        first_time = valid_time[risk_mask].iloc[0]
        last_time = valid_time[risk_mask].iloc[-1]
        logger.info(
            "[icing] %s – %d periods with risk>=4, risk>=1 from %s to %s",
            model_name,
            high_count,
            first_time,
            last_time,
        )
    else:
        logger.info("[icing] %s – no icing risk detected.", model_name)


def log_valid_time_summary(model_name: str, run_date: dt.date, cycle: int, df: pd.DataFrame) -> None:
    """Log the valid time coverage and timestep for a model output."""

    if "valid_time" not in df.columns or df["valid_time"].empty:
        return
    times = df["valid_time"]
    first = times.iloc[0]
    last = times.iloc[-1]
    diffs = times.diff().dt.total_seconds().dropna() / 3600.0
    step = float(diffs.median()) if not diffs.empty else float("nan")
    logger.info(
        "[time] %s – run %s t%02dz valid %s to %s (step≈%.2fh, %d periods)",
        model_name,
        run_date.strftime("%Y%m%d"),
        cycle,
        first,
        last,
        step,
        len(times),
    )


# ---------------------------------------------------------------------------
# GFS implementation
# ---------------------------------------------------------------------------


class GFSPointForecast(PointModelForecast):
    def __init__(self, run_date: dt.date, cycle: int, lat: float, lon: float, fhours: list[int], work_dir: Path):
        self.run_date = run_date
        self.cycle = cycle
        self.lat = lat
        self.lon = lon
        self.fhours = fhours
        self.work_dir = work_dir

    @property
    def model_name(self) -> str:  # pragma: no cover - trivial
        return "GFS"

    def fetch(self, *, no_cache: bool = False) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        for fh in self.fhours:
            url = build_gfs_url(self.run_date, self.cycle, fh)
            out_path = self.work_dir / f"gfs.t{self.cycle:02d}z.f{fh:03d}.grib2"
            download_grib(url, out_path, no_cache=no_cache)

    def to_dataframe(self, *, include_raw_snow_vars: bool = False) -> pd.DataFrame:
        records = []

        for fh in self.fhours:
            path = self.work_dir / f"gfs.t{self.cycle:02d}z.f{fh:03d}.grib2"
            if not path.exists():
                raise FileNotFoundError(f"Missing GRIB file: {path}. Run fetch() first.")

            record = extract_surface_record(path, self.lat, self.lon, fh)
            records.append(record)

        return finalize_point_dataframe(records, include_raw_snow_vars=include_raw_snow_vars)


class AIGFSPointForecast(PointModelForecast):
    def __init__(self, run_date: dt.date, cycle: int, lat: float, lon: float, fhours: list[int], work_dir: Path):
        self.run_date = run_date
        self.cycle = cycle
        self.lat = lat
        self.lon = lon
        self.fhours = fhours
        self.work_dir = work_dir

    @property
    def model_name(self) -> str:  # pragma: no cover - trivial
        return "AIGFS"

    def fetch(self, *, no_cache: bool = False) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        downloaded_any = False
        for fh in self.fhours:
            out_path = self.work_dir / f"aigfs.t{self.cycle:02d}z.sfc.f{fh:03d}.grib2"
            last_exc: requests.HTTPError | None = None
            for base_url in AIGFS_BASE_URLS:
                url = build_aigfs_sfc_url(self.run_date, self.cycle, fh, base_url=base_url)
                try:
                    download_grib(url, out_path, no_cache=no_cache)
                    last_exc = None
                    downloaded_any = True
                    break
                except requests.HTTPError as exc:
                    last_exc = exc
                    if _is_missing_http_error(exc):
                        continue
                    raise
            if last_exc is not None:
                print(f"[download] AIGFS f{fh:03d} unavailable ({last_exc}); skipping this hour.")
        if not downloaded_any:
            raise FileNotFoundError("AIGFS download skipped all requested hours.")

    def to_dataframe(self, *, include_raw_snow_vars: bool = False) -> pd.DataFrame:
        records = []

        for fh in self.fhours:
            path = self.work_dir / f"aigfs.t{self.cycle:02d}z.sfc.f{fh:03d}.grib2"
            if not path.exists():
                print(f"[warn] Missing AIGFS GRIB file for f{fh:03d}: {path.name}")
                continue

            record = extract_surface_record(path, self.lat, self.lon, fh)
            records.append(record)

        if not records:
            raise FileNotFoundError("AIGFS data unavailable for requested hours.")

        return finalize_point_dataframe(records, include_raw_snow_vars=include_raw_snow_vars)


class HGEFSMeanPointForecast(PointModelForecast):
    def __init__(self, run_date: dt.date, cycle: int, lat: float, lon: float, fhours: list[int], work_dir: Path):
        self.run_date = run_date
        self.cycle = cycle
        self.lat = lat
        self.lon = lon
        self.fhours = fhours
        self.work_dir = work_dir

    @property
    def model_name(self) -> str:  # pragma: no cover - trivial
        return "HGEFS-mean"

    def fetch(self, *, no_cache: bool = False) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        downloaded_any = False
        for fh in self.fhours:
            avg_path = self.work_dir / f"hgefs.t{self.cycle:02d}z.sfc.avg.f{fh:03d}.grib2"
            spr_path = self.work_dir / f"hgefs.t{self.cycle:02d}z.sfc.spr.f{fh:03d}.grib2"
            last_avg_exc: requests.HTTPError | None = None
            for base_url in HGEFS_BASE_URLS:
                avg_url = build_hgefs_sfc_avg_url(self.run_date, self.cycle, fh, base_url=base_url)
                try:
                    download_grib(avg_url, avg_path, no_cache=no_cache)
                    last_avg_exc = None
                    downloaded_any = True
                    break
                except requests.HTTPError as exc:
                    last_avg_exc = exc
                    if _is_missing_http_error(exc):
                        continue
                    raise
            if last_avg_exc is not None:
                print(f"[download] HGEFS-mean avg f{fh:03d} unavailable ({last_avg_exc}); skipping this hour.")
                continue

            spr_downloaded = False
            for base_url in HGEFS_BASE_URLS:
                spr_url = build_hgefs_sfc_spr_url(self.run_date, self.cycle, fh, base_url=base_url)
                try:
                    download_grib(spr_url, spr_path, no_cache=no_cache)
                    spr_downloaded = True
                    break
                except requests.HTTPError as exc:
                    if _is_missing_http_error(exc):
                        continue
                    print(f"[download] Optional HGEFS spread missing for f{fh:03d}: {exc}")
                    break
            if not spr_downloaded and spr_path.exists():
                spr_path.unlink(missing_ok=True)

        if not downloaded_any:
            raise FileNotFoundError("HGEFS-mean download skipped all requested hours.")

    def to_dataframe(self, *, include_raw_snow_vars: bool = False) -> pd.DataFrame:
        records = []

        for fh in self.fhours:
            avg_path = self.work_dir / f"hgefs.t{self.cycle:02d}z.sfc.avg.f{fh:03d}.grib2"
            if not avg_path.exists():
                print(f"[warn] Missing HGEFS-mean GRIB file for f{fh:03d}: {avg_path.name}")
                continue

            record = extract_surface_record(avg_path, self.lat, self.lon, fh)

            spr_path = self.work_dir / f"hgefs.t{self.cycle:02d}z.sfc.spr.f{fh:03d}.grib2"
            if spr_path.exists():
                spread_values = extract_hgefs_spread_record(spr_path, self.lat, self.lon)
                record.update(spread_values)

            records.append(record)

        if not records:
            raise FileNotFoundError("HGEFS-mean data unavailable for requested hours.")

        return finalize_point_dataframe(records, include_raw_snow_vars=include_raw_snow_vars)


class NAMPointForecast(PointModelForecast):
    def __init__(self, run_date: dt.date, cycle: int, lat: float, lon: float, fhours: list[int], work_dir: Path):
        self.run_date = run_date
        self.cycle = cycle
        self.lat = lat
        self.lon = lon
        self.fhours = fhours
        self.work_dir = work_dir

    @property
    def model_name(self) -> str:
        return "NAM"

    def fetch(self, *, no_cache: bool = False) -> None:
        """
        Download NAM surface files for the requested forecast hours.

        Special behavior for very fresh runs:
        - If none of the earliest requested hours are available (e.g., 0, 3, 6),
          treat this cycle as unavailable so the caller can fall back.
        """

        self.work_dir.mkdir(parents=True, exist_ok=True)

        downloaded_any = False
        successful_hours: list[int] = []

        for fh in self.fhours:
            url = build_nam_url(self.run_date, self.cycle, fh)
            out_path = self.work_dir / f"nam.t{self.cycle:02d}z.f{fh:03d}.grib2"

            try:
                download_grib(url, out_path, no_cache=no_cache)
                downloaded_any = True
                successful_hours.append(fh)
            except requests.HTTPError as exc:
                if _is_missing_http_error(exc):
                    logger.warning("[NAM] Missing f%03d", fh)
                    continue
                raise

        if not downloaded_any:
            raise FileNotFoundError("NAM download failed for all requested hours.")

        earliest_requested = sorted([fh for fh in self.fhours if fh <= 84])
        if earliest_requested:
            critical_hours = earliest_requested[:3]
            if not any(h in successful_hours for h in critical_hours):
                logger.info(
                    "[NAM] Earliest hours %s unavailable for %s t%02dz; treating cycle as not ready.",
                    critical_hours,
                    self.run_date.strftime("%Y%m%d"),
                    self.cycle,
                )
                raise FileNotFoundError("NAM earliest forecast hours missing; cycle not ready.")

    def to_dataframe(self, *, include_raw_snow_vars: bool = False) -> pd.DataFrame:
        records: list[dict] = []
        for fh in self.fhours:
            path = self.work_dir / f"nam.t{self.cycle:02d}z.f{fh:03d}.grib2"
            if not path.exists():
                continue
            records.append(extract_surface_record(path, self.lat, self.lon, fh))
        if not records:
            raise FileNotFoundError("NAM data unavailable for requested hours.")
        return finalize_point_dataframe(records, include_raw_snow_vars=include_raw_snow_vars)


class HRRRPointForecast(PointModelForecast):
    def __init__(self, run_date: dt.date, cycle: int, lat: float, lon: float, fhours: list[int], work_dir: Path):
        self.run_date = run_date
        self.cycle = cycle
        self.lat = lat
        self.lon = lon
        self.fhours = fhours
        self.work_dir = work_dir

    @property
    def model_name(self) -> str:
        return "HRRR"

    def fetch(self, *, no_cache: bool = False) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        downloaded = False
        for fh in self.fhours:
            url = build_hrrr_url(self.run_date, self.cycle, fh)
            out_path = self.work_dir / f"hrrr.t{self.cycle:02d}z.f{fh:02d}.grib2"
            try:
                download_grib(url, out_path, no_cache=no_cache)
                downloaded = True
            except requests.HTTPError as exc:
                if _is_missing_http_error(exc):
                    logger.warning("[HRRR] Missing f%02d", fh)
                    continue
                raise
        if not downloaded:
            raise FileNotFoundError("HRRR download failed for all requested hours.")

    def to_dataframe(self, *, include_raw_snow_vars: bool = False) -> pd.DataFrame:
        records: list[dict] = []
        for fh in self.fhours:
            path = self.work_dir / f"hrrr.t{self.cycle:02d}z.f{fh:02d}.grib2"
            if not path.exists():
                continue
            records.append(extract_surface_record(path, self.lat, self.lon, fh))
        if not records:
            raise FileNotFoundError("HRRR data unavailable for requested hours.")
        return finalize_point_dataframe(records, include_raw_snow_vars=include_raw_snow_vars)


class RAPPointForecast(PointModelForecast):
    def __init__(self, run_date: dt.date, cycle: int, lat: float, lon: float, fhours: list[int], work_dir: Path):
        self.run_date = run_date
        self.cycle = cycle
        self.lat = lat
        self.lon = lon
        self.fhours = fhours
        self.work_dir = work_dir

    @property
    def model_name(self) -> str:
        return "RAP"

    def fetch(self, *, no_cache: bool = False) -> None:
        """
        Download RAP surface files for the requested hours.

        RAP typically runs to about 39 hours; we cap there for safety.
        """

        self.work_dir.mkdir(parents=True, exist_ok=True)
        downloaded_any = False

        for fh in self.fhours:
            url = build_rap_url(self.run_date, self.cycle, fh)
            out_path = self.work_dir / f"rap.t{self.cycle:02d}z.f{fh:03d}.grib2"

            try:
                download_grib(url, out_path, no_cache=no_cache)
                downloaded_any = True
            except requests.HTTPError as exc:
                if _is_missing_http_error(exc):
                    logger.warning("[RAP] Missing f%03d", fh)
                    continue
                raise

        if not downloaded_any:
            raise FileNotFoundError("RAP download failed for all requested hours.")

    def to_dataframe(self, *, include_raw_snow_vars: bool = False) -> pd.DataFrame:
        records: list[dict] = []

        for fh in self.fhours:
            path = self.work_dir / f"rap.t{self.cycle:02d}z.f{fh:03d}.grib2"
            if not path.exists():
                continue
            records.append(extract_surface_record(path, self.lat, self.lon, fh))

        if not records:
            raise FileNotFoundError("RAP data unavailable for requested hours.")

        return finalize_point_dataframe(records, include_raw_snow_vars=include_raw_snow_vars)


class GEFSControlPointForecast(PointModelForecast):
    def __init__(self, run_date: dt.date, cycle: int, lat: float, lon: float, fhours: list[int], work_dir: Path):
        self.run_date = run_date
        self.cycle = cycle
        self.lat = lat
        self.lon = lon
        self.fhours = fhours
        self.work_dir = work_dir

    @property
    def model_name(self) -> str:
        return "GEFS-control"

    def fetch(self, *, no_cache: bool = False) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        downloaded = False
        for fh in self.fhours:
            url = build_gefs_control_url(self.run_date, self.cycle, fh)
            out_path = self.work_dir / f"gefs_ctrl.t{self.cycle:02d}z.f{fh:03d}.grib2"
            try:
                download_grib(url, out_path, no_cache=no_cache)
                downloaded = True
            except requests.HTTPError as exc:
                if _is_missing_http_error(exc):
                    logger.warning("[GEFS-control] Missing f%03d", fh)
                    continue
                raise
        if not downloaded:
            raise FileNotFoundError("GEFS control download failed for requested hours.")

    def to_dataframe(self, *, include_raw_snow_vars: bool = False) -> pd.DataFrame:
        records: list[dict] = []
        for fh in self.fhours:
            path = self.work_dir / f"gefs_ctrl.t{self.cycle:02d}z.f{fh:03d}.grib2"
            if not path.exists():
                continue
            records.append(extract_surface_record(path, self.lat, self.lon, fh))
        if not records:
            raise FileNotFoundError("GEFS control data unavailable for requested hours.")
        return finalize_point_dataframe(records, include_raw_snow_vars=include_raw_snow_vars)


class GEFSEnsMeanPointForecast(PointModelForecast):
    def __init__(self, run_date: dt.date, cycle: int, lat: float, lon: float, fhours: list[int], work_dir: Path):
        self.run_date = run_date
        self.cycle = cycle
        self.lat = lat
        self.lon = lon
        self.fhours = fhours
        self.work_dir = work_dir

    @property
    def model_name(self) -> str:
        return "GEFS-mean"

    def fetch(self, *, no_cache: bool = False) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        downloaded = False
        for fh in self.fhours:
            url = build_gefs_mean_url(self.run_date, self.cycle, fh)
            out_path = self.work_dir / f"gefs_mean.t{self.cycle:02d}z.f{fh:03d}.grib2"
            try:
                download_grib(url, out_path, no_cache=no_cache)
                downloaded = True
            except requests.HTTPError as exc:
                if _is_missing_http_error(exc):
                    logger.warning("[GEFS-mean] Missing f%03d", fh)
                    continue
                raise
        if not downloaded:
            raise FileNotFoundError("GEFS mean download failed for requested hours.")

    def to_dataframe(self, *, include_raw_snow_vars: bool = False) -> pd.DataFrame:
        records: list[dict] = []
        for fh in self.fhours:
            path = self.work_dir / f"gefs_mean.t{self.cycle:02d}z.f{fh:03d}.grib2"
            if not path.exists():
                continue
            records.append(extract_surface_record(path, self.lat, self.lon, fh))
        if not records:
            raise FileNotFoundError("GEFS mean data unavailable for requested hours.")
        return finalize_point_dataframe(records, include_raw_snow_vars=include_raw_snow_vars)


MODEL_REGISTRY = {
    "gfs": GFSPointForecast,
    "nam": NAMPointForecast,
    "hrrr": HRRRPointForecast,
    "rap": RAPPointForecast,
    "gefs": GEFSControlPointForecast,
    "gefs-mean": GEFSEnsMeanPointForecast,
    "aigfs": AIGFSPointForecast,
    "hgefs": HGEFSMeanPointForecast,
}
MODEL_COLOR_OVERRIDES = {
    "RAP": "#00cc44",
    "HRRR": "#d62728",
    "NAM": "#1f77b4",
}

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_meteogram(model_dfs: dict[str, pd.DataFrame], location_label: str, run_label: str, out_path: Path) -> None:
    if not model_dfs:
        raise ValueError("No model data provided for plotting.")

    has_mslp = any(
        ("mslp_hpa" in df.columns and df["mslp_hpa"].notna().any())
        for df in model_dfs.values()
    )
    n_panels = 4 + int(has_mslp)

    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(14, 3 * n_panels),
        sharex=True,
        constrained_layout=True,
    )

    all_times = pd.concat([df["valid_time"] for df in model_dfs.values()])
    x_min = all_times.min()
    x_max = all_times.max()

    cmap = colormaps.get_cmap("tab10")
    model_styles: dict[str, dict] = {}
    for idx, name in enumerate(model_dfs.keys()):
        color = MODEL_COLOR_OVERRIDES.get(name.upper(), cmap(idx % cmap.N))
        model_styles[name] = {"color": color, "linestyle": "-"}

    ax_temp = axes[0]
    for name, df in model_dfs.items():
        times = df["valid_time"]
        style = model_styles[name]
        ax_temp.plot(times, df["temp_f"], color=style["color"], linestyle=style["linestyle"])
        if "apparent_temp_f" in df.columns:
            ax_temp.plot(
                times,
                df["apparent_temp_f"],
                color=style["color"],
                linestyle="--",
                alpha=0.7,
            )
    ax_temp.axhline(32.0, color="k", linestyle=":", linewidth=1)
    ax_temp.set_ylabel("Temperature (°F)")
    ax_temp.grid(True, alpha=0.3)

    ax_qpf = axes[1]
    ptype_y_offset = 0.05
    markers = {"snow": "*", "mix": "s", "rain": "o", "none": "x"}
    for name, df in model_dfs.items():
        times = df["valid_time"]
        style = model_styles[name]
        median_step = df["valid_time"].diff().dropna().median()
        if pd.isna(median_step):
            median_step = pd.Timedelta(hours=1)
        width = (median_step / pd.Timedelta(days=1)) * 0.8
        ax_qpf.bar(
            times,
            df["qpf_in"],
            width=width,
            color=style["color"],
            alpha=0.3,
            align="center",
        )
        if "precip_type" in df.columns:
            for ptype, marker in markers.items():
                mask = df["precip_type"] == ptype
                if mask.any():
                    ax_qpf.scatter(
                        times[mask],
                        np.full(mask.sum(), ptype_y_offset),
                        marker=marker,
                        color=style["color"],
                        alpha=0.7,
                    )
    ax_qpf.set_ylabel("QPF (in / period)")
    ax_qpf.grid(True, alpha=0.3)

    ax_snow = axes[2]
    for name, df in model_dfs.items():
        times = df["valid_time"]
        style = model_styles[name]
        if "snowfall_in" in df.columns:
            median_step = df["valid_time"].diff().dropna().median()
            if pd.isna(median_step):
                median_step = pd.Timedelta(hours=1)
            width = (median_step / pd.Timedelta(days=1)) * 0.6
            ax_snow.bar(
                times,
                df["snowfall_in"],
                width=width,
                color=style["color"],
                alpha=0.25,
            )
        if "snow_acc_in" in df.columns:
            ax_snow.plot(times, df["snow_acc_in"], color=style["color"], linestyle="-")
    ax_snow.set_ylabel("Snowfall / Snow Depth (in)")
    ax_snow.grid(True, alpha=0.3)

    ax_wind = axes[3]
    for name, df in model_dfs.items():
        times = df["valid_time"]
        style = model_styles[name]
        if "wind10m_mph" in df.columns:
            ax_wind.plot(times, df["wind10m_mph"], color=style["color"])
        if "gust_mph" in df.columns and df["gust_mph"].notna().any():
            ax_wind.plot(times, df["gust_mph"], color=style["color"], linestyle="--", alpha=0.7)
    ax_wind.set_ylabel("Wind (mph)")
    ax_wind.grid(True, alpha=0.3)

    if has_mslp:
        ax_p = axes[4]
        for name, df in model_dfs.items():
            if "mslp_hpa" not in df.columns or df["mslp_hpa"].isna().all():
                continue
            style = model_styles[name]
            ax_p.plot(df["valid_time"], df["mslp_hpa"], color=style["color"])
        ax_p.set_ylabel("MSLP (hPa)")
        ax_p.grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    model_handles = [
        plt.Line2D([0], [0], color=style["color"], linestyle="-", label=name)
        for name, style in model_styles.items()
    ]
    axes[0].legend(handles=model_handles, title="Models", loc="upper left")

    fig.autofmt_xdate()
    model_list_str = ", ".join(model_dfs.keys())
    fig.suptitle(
        f"Meteogram – {location_label} – {run_label} – Models: {model_list_str}",
        fontsize=14,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved meteogram to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_hours_range(hours_arg: str) -> list[int]:
    """Parse hours string like "0-120" or "0-72:6" into a list of ints."""

    if ":" in hours_arg:
        range_part, step_part = hours_arg.split(":", 1)
        step = int(step_part)
    else:
        range_part = hours_arg
        step = 3

    start_str, end_str = range_part.split("-", 1)
    start = int(start_str)
    end = int(end_str)
    return list(range(start, end + 1, step))


def main():
    env_lat = get_env_float("WXGRAPH_LAT", DEFAULT_LAT)
    env_lon = get_env_float("WXGRAPH_LON", DEFAULT_LON)
    env_label = get_env_str("WXGRAPH_SITE", "KCLE")
    env_outdir = get_env_str("WXGRAPH_OUTPUT_DIR", "gfs_meteogram_output")
    env_models = get_env_list("WXGRAPH_MODELS", ["gfs"])
    env_hours = get_env_str("WXGRAPH_HOURS", "0-168:3")
    env_log_level = get_env_str("WXGRAPH_LOG_LEVEL", LOG_LEVEL)

    parser = argparse.ArgumentParser(description="GFS meteogram for a point location")
    parser.add_argument("--date", help="Run date in YYYYMMDD (UTC). Default: today (UTC, adjusted if needed).")
    parser.add_argument("--cycle", type=int, choices=[0, 6, 12, 18], help="GFS cycle hour (0/6/12/18 UTC). Default: most recent.")
    parser.add_argument("--hours", default=env_hours, help="Forecast hours range, e.g., 0-168 or 0-72:6")
    parser.add_argument("--lat", type=float, default=env_lat, help="Latitude for point forecast")
    parser.add_argument("--lon", type=float, default=env_lon, help="Longitude for point forecast")
    parser.add_argument("--location-label", default=env_label, help="Label for plots/output")
    parser.add_argument("--outdir", default=env_outdir, help="Directory to store GRIB files and plot")
    parser.add_argument("--no-cache", action="store_true", help="Force re-download of GRIB files")
    parser.add_argument("--write-csv", action="store_true", help="Write the derived dataframe to CSV alongside the plot")
    parser.add_argument("--merge-csv", action="store_true", help="Write merged CSV of model data.")
    parser.add_argument(
        "--write-json",
        action="store_true",
        help="Write merged meteogram output as JSON for web visualization.",
    )
    parser.add_argument(
        "--models",
        default=",".join(env_models),
        help=(
            "Comma-separated list of models to include. "
            "Supported: gfs, nam, hrrr, rap, gefs, gefs-mean, aigfs, hgefs. "
            "Example: --models=gfs,nam,rap,gefs,aigfs,hgefs"
        ),
    )
    parser.add_argument(
        "--log-level",
        default=env_log_level,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )

    args = parser.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
    now_utc = dt.datetime.now(dt.timezone.utc)

    if args.date:
        run_date = dt.datetime.strptime(args.date, "%Y%m%d").date()
    else:
        run_date = now_utc.date()

    if args.cycle is not None:
        cycle = args.cycle
    else:
        cycle = choose_default_cycle(now_utc)
        if now_utc.hour < cycle:
            run_date = run_date - dt.timedelta(days=1)

    hours = parse_hours_range(args.hours)

    if not args.date:
        latest = find_latest_available_gfs_run(run_date, cycle)
        if latest is not None and latest != (run_date, cycle):
            run_date, cycle = latest
            logger.info(
                "[init] Using available GFS run %s t%02dz",
                run_date.strftime("%Y%m%d"),
                cycle,
            )

    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    if not models:
        models = ["gfs"]

    supported = set(MODEL_REGISTRY.keys())
    unknown = [m for m in models if m not in supported]
    if unknown:
        raise ValueError(f"Unsupported model(s): {', '.join(unknown)}. Supported: {', '.join(sorted(supported))}.")

    max_backtrack = 8
    attempts = 0
    model_dfs: dict[str, pd.DataFrame] = {}
    skipped_for_missing: list[str] = []

    while True:
        work_dir = Path(args.outdir) / f"gfs_{run_date.strftime('%Y%m%d')}_t{cycle:02d}z"
        work_dir.mkdir(parents=True, exist_ok=True)
        plot_path = work_dir / f"meteogram_{args.location_label}_{run_date.strftime('%Y%m%d')}_t{cycle:02d}z.png"

        forecast_entries: list[PointModelForecast] = []
        for model_key in models:
            model_cls = MODEL_REGISTRY[model_key]
            model_hours = limit_forecast_hours(model_key, hours)
            if not model_hours:
                continue
            forecast_entries.append(
                model_cls(run_date, cycle, args.lat, args.lon, model_hours, work_dir / model_key)
            )

        model_dfs.clear()
        skipped_for_missing.clear()
        errors: dict[str, Exception] = {}

        for forecast in forecast_entries:
            name = forecast.model_name
            try:
                forecast.fetch(no_cache=args.no_cache)
                df_raw = forecast.to_dataframe(include_raw_snow_vars=True)
                df = validate_df(df_raw, name)
                df = df.drop(columns=["snowfall_native_in_raw", "snow_depth_native_in"], errors="ignore")
                df = add_road_icing_diagnostics(df)
                log_road_icing_summary(name, df)
                log_valid_time_summary(name, run_date, cycle, df)
            except requests.HTTPError as exc:
                errors[name] = exc
                if _is_missing_http_error(exc):
                    skipped_for_missing.append(name)
                    print(f"[warn] {name} run {run_date.strftime('%Y%m%d')} t{cycle:02d}z not found; skipping.")
                    continue
                raise
            except FileNotFoundError as exc:
                errors[name] = exc
                skipped_for_missing.append(name)
                print(f"[warn] {name} files missing after download attempt: {exc}")
                continue
            model_dfs[name] = df

        if model_dfs:
            break

        # No models succeeded. Determine whether to backtrack to previous cycle.
        first_error = next(iter(errors.values()), None)
        is_404 = any(
            (isinstance(err, requests.HTTPError) and _is_missing_http_error(err))
            or isinstance(err, FileNotFoundError)
            for err in errors.values()
        )
        if args.date or not is_404:
            if first_error:
                raise first_error
            raise RuntimeError("No model data available for requested configuration.")

        attempts += 1
        if attempts >= max_backtrack:
            raise RuntimeError("Unable to locate any available model runs after checking previous cycles.")

        print(
            f"[info] Falling back to previous cycle because run {run_date.strftime('%Y%m%d')} t{cycle:02d}z is unavailable."
        )
        run_date, cycle = step_back_cycle(run_date, cycle)
        continue

    run_label = f"{run_date.strftime('%Y-%m-%d')} {cycle:02d}Z"
    compare_models(model_dfs)
    plot_meteogram(model_dfs, args.location_label, run_label, plot_path)

    if args.write_csv:
        for name, df in model_dfs.items():
            csv_path = plot_path.with_name(
                f"meteogram_{args.location_label}_{run_date.strftime('%Y%m%d')}_t{cycle:02d}z_{name}.csv"
            )
            df.to_csv(csv_path, index=False)
            print(f"[csv] Saved {name} DataFrame to {csv_path}")

    if args.merge_csv:
        merged_path = plot_path.with_name(
            f"meteogram_{args.location_label}_{run_date.strftime('%Y%m%d')}_t{cycle:02d}z_merged.csv"
        )
        merge_model_csvs(model_dfs, merged_path)

    if args.write_json:
        json_path = plot_path.with_name("meteogram_latest.json")
        write_merged_json(model_dfs, json_path)

    if skipped_for_missing:
        skipped_str = ", ".join(skipped_for_missing)
        print(f"[warn] The following models were skipped due to missing data: {skipped_str}")


if __name__ == "__main__":
    main()
