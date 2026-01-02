"""Helpers for opening cfgrib datasets and extracting point records."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import datetime as dt
import logging

import numpy as np
import pandas as pd
import xarray as xr
from cfgrib.dataset import DatasetBuildError

logger = logging.getLogger(__name__)


def get_coord_names(ds: xr.Dataset) -> tuple[str, str]:
    """Identify the latitude/longitude coordinate names in the dataset."""

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


def open_point_dataset(path: Path, lat: float, lon: float, filter_by_keys: dict[str, object]) -> xr.Dataset | None:
    """
    Open a cfgrib dataset filtered by ``filter_by_keys`` and select the nearest point.
    """

    idx_path = Path(f"{path}.idx")

    def _open() -> xr.Dataset:
        return xr.open_dataset(path, engine="cfgrib", backend_kwargs={"filter_by_keys": filter_by_keys})

    try:
        ds = _open()
    except (EOFError, DatasetBuildError) as exc:
        if idx_path.exists():
            logger.warning("Removing stale cfgrib index %s (%s)", idx_path, exc)
            idx_path.unlink()
            ds = _open()
        else:
            raise

    if not ds.data_vars:
        ds.close()
        return None

    for coord in ("latitude", "lat"):
        if coord in ds and coord not in ds.coords:
            ds = ds.assign_coords({coord: ds[coord]})
    for coord in ("longitude", "lon"):
        if coord in ds and coord not in ds.coords:
            ds = ds.assign_coords({coord: ds[coord]})

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
            logger.debug("Point select: scalar grid %s()", path)
            return dataset.load()

        if lat_vals.shape == lon_vals.shape:
            dist = (lat_vals - lat) ** 2 + (lon_vals - lon_target) ** 2
            flat_idx = int(np.nanargmin(dist))
            unraveled = np.unravel_index(flat_idx, lat_vals.shape)
            indexers: dict[str, int] = {}
            for dim, idx_val in zip(lat_da.dims, unraveled):
                indexers[dim] = int(idx_val)
            return dataset.isel(indexers).load()

        if lat_vals.ndim == 1 and lon_vals.ndim == 1:
            idx_lat = int(np.nanargmin(np.abs(lat_vals - lat)))
            idx_lon = int(np.nanargmin(np.abs(lon_vals - lon_target)))
            lat_dim = lat_da.dims[0]
            lon_dim = lon_da.dims[0]
            return dataset.isel({lat_dim: idx_lat, lon_dim: idx_lon}).load()

        raise ValueError("Latitude/longitude coordinate shapes do not match.")

    try:
        point = _nearest_point(ds)
    finally:
        ds.close()

    return point


def c_to_f(temp_c: np.ndarray) -> np.ndarray:
    """Convert Celsius to Fahrenheit."""

    return temp_c * 9.0 / 5.0 + 32.0


def rh_from_t_td(t_c: np.ndarray, td_c: np.ndarray) -> np.ndarray:
    """Calc relative humidity (%) from temperature and dewpoint."""

    es = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
    e = 6.112 * np.exp((17.67 * td_c) / (td_c + 243.5))
    return 100.0 * e / es


def dewpoint_from_spfh(temp_c: np.ndarray, spfh: np.ndarray, pressure_hpa: float = 1013.0) -> np.ndarray:
    """Approximate dewpoint (°C) from specific humidity (kg/kg) and pressure."""

    epsilon = 0.622
    w = spfh / (1.0 - spfh)
    e = pressure_hpa * w / (w + epsilon)
    ln_ratio = np.log(e / 6.112)
    return (243.5 * ln_ratio) / (17.67 - ln_ratio)


def extract_surface_record(
    path: Path,
    lat: float,
    lon: float,
    fh: int,
    *,
    gust_var_candidates: Iterable[str] | None = None,
) -> dict[str, object]:
    """Extract surface-level metadata from the nearest point."""

    gust_candidates = tuple(gust_var_candidates or ("GUST_surface", "GUST"))

    try:
        pt_surface_inst = open_point_dataset(
            path,
            lat,
            lon,
            {"typeOfLevel": "surface", "stepType": "instant"},
        )
    except DatasetBuildError as exc:
        logger.warning("Falling back to generic surface instant read for %s: %s", path, exc)
        pt_surface_inst = open_point_dataset(path, lat, lon, {"typeOfLevel": "surface"})

    pt_surface_acc = None
    try:
        pt_surface_acc = open_point_dataset(
            path,
            lat,
            lon,
            {"typeOfLevel": "surface", "stepType": "accum"},
        )
    except DatasetBuildError:
        logger.debug("Accumulated surface levels not present for %s", path)

    pt_2m = open_point_dataset(path, lat, lon, {"typeOfLevel": "heightAboveGround", "level": 2})
    pt_10m = open_point_dataset(path, lat, lon, {"typeOfLevel": "heightAboveGround", "level": 10})
    pt_mean_sea = open_point_dataset(path, lat, lon, {"typeOfLevel": "meanSea"})
    pt_atmos = None
    pt_entire_atm = None
    try:
        pt_atmos = open_point_dataset(path, lat, lon, {"typeOfLevel": "atmosphere"})
    except (DatasetBuildError, KeyError):
        logger.debug("Atmosphere cloud cover missing for %s", path)
    try:
        pt_entire_atm = open_point_dataset(path, lat, lon, {"typeOfLevel": "entireAtmosphere"})
    except (DatasetBuildError, KeyError):
        logger.debug("Entire-atmosphere cloud cover missing for %s", path)

    def select_from_points(points: list[xr.Dataset | None], candidates: Iterable[str]):
        for pt in points:
            if pt is None:
                continue
            for name in candidates:
                if name in pt.data_vars:
                    return pt[name]
        raise KeyError(f"No variable {candidates} in point datasets")

    temp_k = np.nan
    try:
        t = select_from_points(
            [pt_2m, pt_surface_inst],
            ["t2m", "2t", "TMP_2maboveground", "TMP_surface", "TMP"],
        )
        temp_k = float(t.values)
    except KeyError:
        logger.debug("Temperature fields missing from %s", path)

    temp_c = temp_k - 273.15 if not np.isnan(temp_k) else np.nan

    dew_c = np.nan
    try:
        dew = select_from_points(
            [pt_2m, pt_surface_inst],
            ["d2m", "2d", "DPT_2maboveground", "DPT_surface", "DPT", "DEPR"],
        )
        dew_val = float(dew.values)
        dew_c = dew_val - 273.15 if dew_val > 200.0 else dew_val
    except KeyError:
        logger.debug("Dewpoint variables missing from %s", path)

    spfh_val = np.nan
    try:
        spfh = select_from_points(
            [pt_2m, pt_surface_inst],
            ["q2m", "SPFH_2maboveground", "SPFH_surface", "SPFH"],
        )
        spfh_val = float(spfh.values)
        if np.isnan(dew_c) and not np.isnan(temp_c):
            dew_c = float(dewpoint_from_spfh(np.array([temp_c]), np.array([spfh_val]))[0])
    except (KeyError, ValueError):
        logger.debug("Specific humidity missing from %s", path)

    rh_percent = np.nan
    if not np.isnan(temp_c) and not np.isnan(dew_c):
        rh_percent = float(rh_from_t_td(np.array([temp_c]), np.array([dew_c]))[0])

    u = np.nan
    v = np.nan
    try:
        u = float(select_from_points([pt_surface_inst, pt_10m], ["UGRD_surface", "UGRD", "u10", "10u"]).values)
        v = float(select_from_points([pt_surface_inst, pt_10m], ["VGRD_surface", "VGRD", "v10", "10v"]).values)
    except KeyError:
        logger.debug("Wind components missing from %s", path)

    wind_ms = float(np.hypot(u, v)) if not (np.isnan(u) or np.isnan(v)) else np.nan
    wind_mph = float(wind_ms * 2.23694) if not np.isnan(wind_ms) else np.nan

    gust_mph = np.nan
    try:
        gust = float(select_from_points([pt_surface_inst, pt_10m], gust_candidates).values)
        gust_mph = gust * 2.23694
    except KeyError:
        logger.debug("Gust variable missing for %s", path)

    qpf_in_raw = 0.0
    try:
        apcp = select_from_points(
            [pt_surface_acc, pt_surface_inst],
            ["APCP_surface", "APCP", "tp", "PRATE_surface"],
        )
        qpf_in_raw = float(apcp.values) / 25.4
    except KeyError:
        logger.debug("Precipitation accumulation missing for %s", path)

    mslp = np.nan
    try:
        prmsl = select_from_points([pt_surface_inst, pt_mean_sea], ["PRMSL", "PRMSL_meansealevel", "prmsl"])
        pr_val = float(prmsl.values)
        mslp = pr_val / 100.0 if pr_val > 2000.0 else pr_val
    except KeyError:
        logger.debug("Pressure variables missing for %s", path)

    cloud_pct = np.nan
    try:
        cloud = select_from_points(
            [pt_surface_inst, pt_entire_atm, pt_atmos, pt_mean_sea],
            ["TCDC_entireatmosphere", "TCDC", "tcc", "TCC", "CLCT"],
        )
        cloud_val = float(cloud.values)
        if not np.isnan(cloud_val):
            if 0.0 <= cloud_val <= 1.0:
                cloud_pct = cloud_val * 100.0
            else:
                cloud_pct = cloud_val
            cloud_pct = float(np.clip(cloud_pct, 0.0, 100.0))
    except KeyError:
        logger.debug("Cloud cover variables missing for %s", path)

    snowfall_native_in = np.nan
    snow_depth_native_in = np.nan
    try:
        snow = select_from_points(
            [pt_surface_acc, pt_surface_inst],
            ["ASNOW_surface", "ASNOW", "WEASD_surface", "WEASD", "SNOW_surface"],
        )
        snowfall_native_in = float(snow.values) / 25.4
    except KeyError:
        logger.debug("Snowfall variable missing for %s", path)

    try:
        snowd = select_from_points([pt_surface_inst, pt_surface_acc], ["SNOD_surface", "SNOD", "snowh"])
        snow_depth_native_in = float(snowd.values) * 39.3701
    except KeyError:
        logger.debug("Snow depth missing for %s", path)

    point_ref = pt_surface_inst or pt_surface_acc or pt_2m or pt_10m or pt_mean_sea
    if point_ref is None:
        raise FileNotFoundError(f"No data could be read from {path}")

    time = point_ref["time"]
    if "step" in point_ref.coords:
        valid_time = pd.to_datetime(time.values) + pd.to_timedelta(point_ref["step"].values)
    else:
        valid_time = pd.to_datetime(time.values) + pd.to_timedelta(fh, "h")

    return {
        "valid_time": pd.to_datetime(valid_time),
        "f_hour": fh,
        "temp_c": temp_c,
        "temp_f": c_to_f(np.array([temp_c]))[0] if not np.isnan(temp_c) else np.nan,
        "tdew_c": dew_c,
        "rh_percent": rh_percent,
        "qpf_in_raw": qpf_in_raw,
        "snowfall_native_in_raw": snowfall_native_in,
        "snow_depth_native_in": snow_depth_native_in,
        "wind10m_ms": wind_ms,
        "wind10m_mph": wind_mph,
        "u10m_ms": u,
        "v10m_ms": v,
        "gust_mph": gust_mph,
        "mslp_hpa": mslp,
        "cloud_pct": cloud_pct,
        "spfh_2m": spfh_val,
    }
