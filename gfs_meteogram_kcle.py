#!/usr/bin/env python3
"""
Phase 1: GFS-only meteogram for KCLE (Cleveland).

What it does:
- Figures out today's UTC date & a recent GFS cycle (00/06/12/18).
- Downloads GFS 0.25° TMP(2 m) + APCP(sfc) around KCLE via NOMADS grib filter.
- Extracts nearest gridpoint to KCLE.
- Computes:
    * 6-hourly precip ("QPF")
    * Approx snowfall (using temp-dependent snow ratio)
    * Accumulated snow with a simple compaction/melt model
- Produces a 3-panel meteogram PNG: snow / precip / temp.

You can override date/cycle/hours via CLI args.
"""

import argparse
import datetime as dt
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr
import matplotlib.pyplot as plt

# ------------------------------
# Config
# ------------------------------

BASE_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"

# Rough KCLE coordinates
KCLE_LAT = 41.48
KCLE_LON = -81.81

# Small lat/lon box around KCLE for the grib filter (degrees)
LAT_TOP = 43.0
LAT_BOTTOM = 40.0
LON_LEFT = -84.0
LON_RIGHT = -79.0


# ------------------------------
# Utility functions
# ------------------------------

def choose_default_cycle(now_utc: dt.datetime) -> int:
    """
    Choose most recent GFS cycle (0, 6, 12, 18 UTC) at or before current UTC.
    """
    for c in [18, 12, 6, 0]:
        if now_utc.hour >= c:
            return c
    return 18  # previous day fallback (we'll also adjust date below)


def build_gfs_url(run_date: dt.date, cycle: int, f_hour: int) -> str:
    """
    Build the NOMADS grib-filter URL for GFS 0.25° with 2m temp + APCP.
    """
    ymd = run_date.strftime("%Y%m%d")
    file_name = f"gfs.t{cycle:02d}z.pgrb2.0p25.f{f_hour:03d}"

    params = {
        "file": file_name,
        # vars
        "lev_2_m_above_ground": "on",
        "lev_surface": "on",
        "var_TMP": "on",
        "var_APCP": "on",
        # subregion
        "subregion": "",
        "leftlon": str(LON_LEFT),
        "rightlon": str(LON_RIGHT),
        "toplat": str(LAT_TOP),
        "bottomlat": str(LAT_BOTTOM),
        # directory
        "dir": f"/gfs.{ymd}/{cycle:02d}/atmos",
    }

    # requests will handle the querystring encoding
    return requests.Request("GET", BASE_URL, params=params).prepare().url


def download_grib_if_needed(url: str, out_path: Path) -> None:
    """
    Download the GRIB file at url to out_path if it doesn't exist.
    """
    if out_path.exists():
        print(f"[download] Using cached {out_path.name}")
        return

    print(f"[download] Fetching {out_path.name}")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)


def get_coord_names(ds: xr.Dataset):
    """
    Try to find the latitude/longitude coordinate names in a cfgrib dataset.
    """
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


def select_var(ds: xr.Dataset, candidates):
    """
    Pick the first variable name in 'candidates' that exists in ds.
    This gives us some robustness across slightly different naming conventions.
    """
    for name in candidates:
        if name in ds.data_vars:
            return ds[name]
    raise KeyError(f"None of {candidates} found in dataset vars: {list(ds.data_vars)}")


def c_to_f(temp_c: np.ndarray) -> np.ndarray:
    return temp_c * 9.0 / 5.0 + 32.0


def simple_snow_ratio(temp_c: float) -> float:
    """
    Very simple temp-dependent snow ratio (snow:water).
    """
    if temp_c <= -10.0:
        return 20.0
    if temp_c <= -5.0:
        return 15.0
    if temp_c <= 0.0:
        return 10.0
    if temp_c <= 1.0:
        return 8.0
    return 0.0  # treat as rain


def compute_snow_and_compaction(qpf: np.ndarray, temp_c: np.ndarray):
    """
    Given per-period QPF (inches) and temp (°C), compute:
      - snowfall per period (inches)
      - accumulated snow depth with crude compaction (inches)
    """
    snowfall = np.zeros_like(qpf)
    snow_acc = np.zeros_like(qpf)

    depth = 0.0
    for i, (p, t) in enumerate(zip(qpf, temp_c)):
        if p > 0:
            r = simple_snow_ratio(t)
            snowfall[i] = p * r
        else:
            snowfall[i] = 0.0

        # compaction / melting when above freezing
        if t > 0.0:
            depth *= 0.96  # lose 4% per period above freezing

        depth += snowfall[i]
        snow_acc[i] = depth

    return snowfall, snow_acc


# ------------------------------
# Core processing
# ------------------------------

def build_timeseries(run_date: dt.date, cycle: int, hours, work_dir: Path):
    """
    Download GFS files for the specified forecast hours and build a time series
    at KCLE for temp + QPF + snow + compacted snow.

    Returns a pandas DataFrame.
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for f_hour in hours:
        url = build_gfs_url(run_date, cycle, f_hour)
        out_path = work_dir / f"gfs.t{cycle:02d}z.f{f_hour:03d}.grib2"
        download_grib_if_needed(url, out_path)

        # Open with cfgrib
        ds = xr.open_dataset(out_path, engine="cfgrib")

        lat_name, lon_name = get_coord_names(ds)
        pt = ds.sel(
            {lat_name: KCLE_LAT, lon_name: KCLE_LON},
            method="nearest"
        )

        # Guess variable names
        t2m = select_var(pt.to_dataset(), ["t2m", "2t", "TMP_2maboveground"])
        apcp = select_var(pt.to_dataset(), ["tp", "prate", "apcp", "APCP_surface"])

        # Time: valid time = analysis time + step (forecast lead)
        time = pt["time"]
        if "step" in pt.coords:
            step = pt["step"]
            valid_time = (time + step).values
        else:
            # Fallback: just add f_hour hours
            valid_time = (pd.to_datetime(time.values) + pd.to_timedelta(f_hour, "h"))

        temp_k = float(t2m.values)  # Kelvin
        temp_c = temp_k - 273.15

        # APCP units are usually kg/m^2 (mm of water). Convert to inches.
        qpf_mm = float(apcp.values)
        qpf_in = qpf_mm / 25.4

        records.append(
            {
                "valid_time": pd.to_datetime(valid_time),
                "f_hour": f_hour,
                "temp_c": temp_c,
                "qpf_in_raw": qpf_in,
            }
        )

    df = pd.DataFrame.from_records(records).sort_values("valid_time")

    # IMPORTANT: For many GFS APCP fields, values are per-interval already.
    # If you discover they are "since model start", uncomment this diff:
    # df["qpf_in"] = df["qpf_in_raw"].diff().clip(lower=0.0).fillna(df["qpf_in_raw"])
    df["qpf_in"] = df["qpf_in_raw"]

    snowfall, snow_acc = compute_snow_and_compaction(
        df["qpf_in"].values, df["temp_c"].values
    )

    df["snowfall_in"] = snowfall
    df["snow_acc_in"] = snow_acc
    df["temp_f"] = c_to_f(df["temp_c"].values)

    return df


def plot_meteogram(df: pd.DataFrame, title_suffix: str, out_path: Path):
    """
    Make a 3-panel meteogram: accumulated snow, QPF, 2m temperature.
    """
    times = df["valid_time"]

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1.5, 2]}
    )

    # Panel 1: Accumulated snow
    ax1.step(times, df["snow_acc_in"], where="mid", linewidth=2)
    ax1.set_ylabel("Snow Depth (in)")
    ax1.grid(True, alpha=0.3)

    # Panel 2: QPF per period
    ax2.bar(times, df["qpf_in"], width=0.15, align="center")
    ax2.set_ylabel("QPF (in / period)")
    ax2.grid(True, alpha=0.3)

    # Panel 3: Temperature
    ax3.plot(times, df["temp_f"], marker="o")
    ax3.axhline(32.0, color="k", linestyle="--", linewidth=1, alpha=0.6)
    ax3.set_ylabel("2 m Temp (°F)")
    ax3.grid(True, alpha=0.3)

    fig.autofmt_xdate()

    fig.suptitle(f"GFS 0.25° Meteogram – KCLE {title_suffix}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[plot] Saved meteogram to {out_path}")


# ------------------------------
# CLI
# ------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: GFS-only meteogram for KCLE."
    )
    parser.add_argument(
        "--date",
        help="Run date in YYYYMMDD (UTC). Default: today (UTC, adjusted if needed).",
    )
    parser.add_argument(
        "--cycle",
        type=int,
        choices=[0, 6, 12, 18],
        help="GFS cycle hour (0/6/12/18 UTC). Default: most recent.",
    )
    parser.add_argument(
        "--fhour-start",
        type=int,
        default=0,
        help="First forecast hour (default 0).",
    )
    parser.add_argument(
        "--fhour-end",
        type=int,
        default=72,
        help="Last forecast hour inclusive (default 72).",
    )
    parser.add_argument(
        "--fhour-step",
        type=int,
        default=6,
        help="Forecast hour step (default 6).",
    )
    parser.add_argument(
        "--outdir",
        default="gfs_meteogram_output",
        help="Directory to store GRIB files and plot.",
    )

    args = parser.parse_args()

    now_utc = dt.datetime.utcnow()

    if args.date:
        run_date = dt.datetime.strptime(args.date, "%Y%m%d").date()
    else:
        run_date = now_utc.date()

    if args.cycle is not None:
        cycle = args.cycle
    else:
        # pick most recent cycle, with simple “previous day if after midnight” tweak
        cycle = choose_default_cycle(now_utc)
        if now_utc.hour < cycle:
            run_date = run_date - dt.timedelta(days=1)

    hours = list(range(args.fhour_start, args.fhour_end + 1, args.fhour_step))

    outdir = Path(args.outdir)
    work_dir = outdir / f"gfs_{run_date.strftime('%Y%m%d')}_t{cycle:02d}z"
    plot_path = work_dir / "kcle_gfs_meteogram.png"

    print(f"Run date: {run_date}  cycle: {cycle:02d}Z  hours: {hours}")
    print(f"Working dir: {work_dir}")

    df = build_timeseries(run_date, cycle, hours, work_dir)
    title_suffix = f"{run_date.strftime('%Y-%m-%d')} {cycle:02d}Z"
    plot_meteogram(df, title_suffix, plot_path)


if __name__ == "__main__":
    main()
