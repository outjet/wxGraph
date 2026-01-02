"""Helper utilities for constructing NOMADS download URLs."""

from __future__ import annotations

from datetime import date
from typing import Mapping, Sequence

import requests

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


def build_gfs_urls(run_date: date, cycle: int, fh: int, bbox: Mapping[str, float]) -> list[str]:
    """Construct a filtered GFS 0.25° URL that honors the provided bbox."""

    ymd = run_date.strftime("%Y%m%d")
    file_name = f"gfs.t{cycle:02d}z.pgrb2.0p25.f{fh:03d}"
    params = {
        "file": file_name,
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
        "subregion": "",
        "leftlon": str(bbox["left"]),
        "rightlon": str(bbox["right"]),
        "toplat": str(bbox["top"]),
        "bottomlat": str(bbox["bottom"]),
        "dir": f"/gfs.{ymd}/{cycle:02d}/atmos",
    }
    request = requests.Request("GET", BASE_URL, params=params).prepare()
    return [request.url]


def build_nam_urls(run_date: date, cycle: int, fh: int, _: Mapping[str, float]) -> list[str]:
    """Return the NOMADS URLs for NAM surface files."""

    ymd = run_date.strftime("%Y%m%d")
    fname = f"nam.t{cycle:02d}z.awphys{fh:02d}.tm00.grib2"
    url = f"{NAM_BASE_URL}/nam.{ymd}/{fname}"
    return [url]


def build_hrrr_urls(run_date: date, cycle: int, fh: int, _: Mapping[str, float]) -> list[str]:
    """Return the NOMADS URLs for HRRR surface files."""

    ymd = run_date.strftime("%Y%m%d")
    fh_str = f"{fh:02d}"
    fname = f"hrrr.t{cycle:02d}z.wrfnatf{fh_str}.grib2"
    url = f"{HRRR_BASE_URL}/hrrr.{ymd}/conus/{fname}"
    return [url]


def build_rap_urls(run_date: date, cycle: int, fh: int, _: Mapping[str, float]) -> list[str]:
    """Return the NOMADS URLs for RAP surface files."""

    ymd = run_date.strftime("%Y%m%d")
    fh_str = f"{fh:02d}"
    fname = f"rap.t{cycle:02d}z.awip32f{fh_str}.grib2"
    url = f"{RAP_BASE_URL}/rap.{ymd}/{fname}"
    return [url]


def build_gefs_urls(run_date: date, cycle: int, fh: int, _: Mapping[str, float]) -> list[str]:
    """Return the NOMADS URLs for GEFS ensemble mean surface files."""

    ymd = run_date.strftime("%Y%m%d")
    fname = f"gefs.t{cycle:02d}z.pgrb2b_mean.f{fh:03d}"
    url = f"{GEFS_BASE_URL}/gefs.{ymd}/{cycle:02d}/ensstat/{fname}"
    return [url]


def build_aigfs_urls(run_date: date, cycle: int, fh: int, _: Mapping[str, float]) -> list[str]:
    """Return the NOMADS URLs for AIGFS deterministic surface fields."""

    ymd = run_date.strftime("%Y%m%d")
    fname = f"aigfs.t{cycle:02d}z.sfc.f{fh:03d}.grib2"
    return [f"{base}/aigfs.{ymd}/{cycle:02d}/model/atmos/grib2/{fname}" for base in AIGFS_BASE_URLS]


def build_hgefs_urls(run_date: date, cycle: int, fh: int, _: Mapping[str, float]) -> list[str]:
    """Return the NOMADS URLs for HGEFS mean surface statistics."""

    ymd = run_date.strftime("%Y%m%d")
    fname = f"hgefs.t{cycle:02d}z.sfc.avg.f{fh:03d}.grib2"
    return [f"{base}/hgefs.{ymd}/{cycle:02d}/ensstat/products/atmos/grib2/{fname}" for base in HGEFS_BASE_URLS]


MODEL_URL_BUILDERS = {
    "gfs": build_gfs_urls,
    "nam": build_nam_urls,
    "hrrr": build_hrrr_urls,
    "rap": build_rap_urls,
    "gefs": build_gefs_urls,
    "aigfs": build_aigfs_urls,
    "hgefs": build_hgefs_urls,
}


def build_nomads_urls(
    model: str,
    run_date: date,
    cycle: int,
    fh: int,
    bbox: Mapping[str, float],
) -> list[str]:
    """
    Return the list of candidate NOMADS URLs for a model/fhour combo.
    """

    builder = MODEL_URL_BUILDERS.get(model.lower())
    if builder is None:
        raise ValueError(f"No NOMADS URL builder defined for {model}")
    return builder(run_date, cycle, fh, bbox)
