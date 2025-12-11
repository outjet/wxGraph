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
from pathlib import Path
from typing import Iterable, Protocol

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import xarray as xr

BASE_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"

# Default KCLE coordinates
DEFAULT_LAT = 41.48
DEFAULT_LON = -81.81

# Small lat/lon box around KCLE for the grib filter (degrees)
LAT_TOP = 43.0
LAT_BOTTOM = 40.0
LON_LEFT = -84.0
LON_RIGHT = -79.0


class PointModelForecast(Protocol):
    def fetch(self, *, no_cache: bool = False) -> None:
        ...

    def to_dataframe(self) -> pd.DataFrame:
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


def download_grib(url: str, out_path: Path, *, no_cache: bool = False) -> None:
    """Download the GRIB file at url to out_path."""

    if out_path.exists() and not no_cache:
        print(f"[download] Using cached {out_path.name}")
        return

    print(f"[download] Fetching {out_path.name}")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)


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


def compute_snowfall_and_compaction(qpf_in: np.ndarray, temp_c: np.ndarray, precip_type: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute snowfall per period and compacted accumulation.
    - Only create snow where precip_type is snow or mix.
    - Snow ratio scales with temperature.
    - Simple temperature-dependent compaction.
    """

    snowfall = np.zeros_like(qpf_in)
    snow_acc = np.zeros_like(qpf_in)
    depth = 0.0

    for i, (qpf, t, ptype) in enumerate(zip(qpf_in, temp_c, precip_type)):
        if ptype in {"snow", "mix"} and qpf > 0:
            if ptype == "mix":
                ratio = 5.0
            elif t <= -5.0:
                ratio = 15.0
            elif t <= 0.0:
                ratio = 12.0
            else:
                ratio = 10.0
            snowfall[i] = qpf * ratio
        else:
            snowfall[i] = 0.0

        if t > 0.0:
            depth *= 0.95  # melt/settle faster above freezing
        else:
            depth *= 0.985  # light settling below freezing

        depth += snowfall[i]
        snow_acc[i] = depth

    return snowfall, snow_acc
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

    def to_dataframe(self) -> pd.DataFrame:
        records = []

        for fh in self.fhours:
            path = self.work_dir / f"gfs.t{self.cycle:02d}z.f{fh:03d}.grib2"
            if not path.exists():
                raise FileNotFoundError(f"Missing GRIB file: {path}. Run fetch() first.")

            # --- Load primarily surface-level fields ---
            ds_sfc = xr.open_dataset(
                path,
                engine="cfgrib",
                backend_kwargs={"filter_by_keys": {"typeOfLevel": "surface"}},
            )
            lat_name, lon_name = get_coord_names(ds_sfc)
            pt_sfc = ds_sfc.sel({lat_name: self.lat, lon_name: self.lon}, method="nearest")

            # --- Temperature: surface first, fall back to 2 m if needed ---
            try:
                t = select_var(pt_sfc, ["TMP_surface"])
                temp_k = float(t.values)
            except KeyError:
                # fallback: 2 m above ground
                ds_2m = xr.open_dataset(
                    path,
                    engine="cfgrib",
                    backend_kwargs={"filter_by_keys": {"typeOfLevel": "heightAboveGround", "level": 2}},
                )
                lat2, lon2 = get_coord_names(ds_2m)
                pt_2m = ds_2m.sel({lat2: self.lat, lon2: self.lon}, method="nearest")
                t = select_var(pt_2m, ["t2m", "2t", "TMP_2maboveground"])
                temp_k = float(t.values)

            temp_c = temp_k - 273.15

            # --- Dewpoint / RH: SPFH_surface if available ---
            try:
                spfh = float(select_var(pt_sfc, ["SPFH_surface", "SPFH"]).values)
                dew_c = float(dewpoint_from_spfh(np.array([temp_c]), np.array([spfh]))[0])
                rh = rh_from_t_td(np.array([temp_c]), np.array([dew_c]))[0]
            except KeyError:
                dew_c = np.nan
                rh = np.nan

            # --- Wind: surface U/V (treat as 10 m equivalent for plotting) ---
            try:
                u = float(select_var(pt_sfc, ["UGRD_surface", "UGRD"]).values)
                v = float(select_var(pt_sfc, ["VGRD_surface", "VGRD"]).values)
            except KeyError:
                u = 0.0
                v = 0.0

            wind_ms = float(np.hypot(u, v))
            wind_mph = wind_ms * 2.23694

            # --- Wind gusts (if present) ---
            try:
                gust = float(select_var(pt_sfc, ["GUST_surface", "GUST"]).values)
                gust_mph = gust * 2.23694
            except KeyError:
                gust_mph = np.nan

            # --- Precipitation: accumulated APCP at surface ---
            apcp = select_var(pt_sfc, ["APCP_surface", "apcp", "tp"])
            qpf_mm = float(apcp.values)
            qpf_in_raw = qpf_mm / 25.4

            # --- Mean Sea Level Pressure (if present) ---
            try:
                prmsl = select_var(pt_sfc, ["PRMSL", "PRMSL_meansealevel"])
                # PRMSL is usually in Pa; convert to hPa if so
                pr_val = float(prmsl.values)
                mslp_hpa = pr_val / 100.0 if pr_val > 2000.0 else pr_val
            except KeyError:
                mslp_hpa = np.nan

            # --- Valid time from GRIB time/step ---
            time = pt_sfc["time"]
            if "step" in pt_sfc.coords:
                valid_time = (
                    pd.to_datetime(time.values)
                    + pd.to_timedelta(pt_sfc["step"].values)
                ).item()
            else:
                valid_time = (
                    pd.to_datetime(time.values)
                    + pd.to_timedelta(fh, "h")
                ).to_pydatetime()

            records.append(
                {
                    "valid_time": pd.to_datetime(valid_time),
                    "f_hour": fh,
                    "temp_c": temp_c,
                    "temp_f": c_to_f(np.array([temp_c]))[0],
                    "tdew_c": dew_c,
                    "rh_percent": rh,
                    "qpf_in_raw": qpf_in_raw,
                    "wind10m_ms": wind_ms,
                    "wind10m_mph": wind_mph,
                    "gust_mph": gust_mph,
                    "mslp_hpa": mslp_hpa,
                }
            )

        # --- Post-processing into a full timeseries ---
        df = pd.DataFrame.from_records(records).sort_values("valid_time").reset_index(drop=True)

        # Convert accumulated APCP to per-period QPF
        df["qpf_in"] = df["qpf_in_raw"].diff().clip(lower=0.0)
        df.loc[df["qpf_in"].isna(), "qpf_in"] = df.loc[df["qpf_in"].isna(), "qpf_in_raw"]

        precip_type = classify_precip_type(df["temp_c"].values, df["qpf_in"].values)
        df["precip_type"] = precip_type

        snowfall, snow_acc = compute_snowfall_and_compaction(
            df["qpf_in"].values, df["temp_c"].values, precip_type
        )
        df["snowfall_in"] = snowfall
        df["snow_acc_in"] = snow_acc

        df["apparent_temp_f"] = apparent_temperature_f(
            df["temp_f"].values,
            df["wind10m_mph"].values,
            df["rh_percent"].fillna(0).values,
        )

        return df

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_meteogram(model_dfs: dict[str, pd.DataFrame], location_label: str, run_label: str, out_path: Path) -> None:
    if not model_dfs:
        raise ValueError("No model data provided for plotting.")

    # For now we only plot the first model provided
    model_name, df = next(iter(model_dfs.items()))
    times = df["valid_time"]

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True, constrained_layout=True)

    has_mslp = "mslp_hpa" in df.columns and df["mslp_hpa"].notna().any()
    n_panels = 5 if has_mslp else 4

    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3 * n_panels), sharex=True, constrained_layout=True)

    # Panel 1: Temperature and apparent temperature
    ax1 = axes[0]
    ax1.plot(times, df["temp_f"], label="2 m Temp (°F)", color="tab:red")
    ax1.plot(times, df["apparent_temp_f"], label="Apparent Temp (°F)", linestyle="--", color="tab:orange")
    ax1.axhline(32.0, color="k", linestyle=":", linewidth=1)
    ax1.set_ylabel("Temperature (°F)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    # Panel 2: QPF and precip type markers
    ax2 = axes[1]
    ax2.bar(times, df["qpf_in"], width=0.02, align="center", color="tab:blue", alpha=0.6, label="QPF (in)")
    ptype_y = np.full_like(df["qpf_in"].values, 0.05)
    colors = {"snow": "blue", "mix": "purple", "rain": "green", "none": "gray"}
    markers = {"snow": "*", "mix": "s", "rain": "o", "none": "x"}
    for ptype in np.unique(df["precip_type"]):
        mask = df["precip_type"] == ptype
        ax2.scatter(times[mask], ptype_y[mask], color=colors.get(ptype, "gray"), marker=markers.get(ptype, "o"), label=f"{ptype.title()} type")
    ax2.set_ylabel("QPF (in / period)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")

    # Panel 3: Snowfall and accumulation
    ax3 = axes[2]
    ax3.bar(times, df["snowfall_in"], width=0.02, align="center", color="tab:cyan", alpha=0.7, label="Snowfall (in)")
    ax3.plot(times, df["snow_acc_in"], color="tab:blue", label="Snow Depth (in)")
    ax3.set_ylabel("Snowfall / Snow Depth (in)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper left")

    # Panel 4: Wind
    ax4 = axes[3]
    ax4.plot(times, df["wind10m_mph"], color="tab:green", label="10 m Wind (mph)")
    ax4.fill_between(times, 25, df["wind10m_mph"], where=df["wind10m_mph"] >= 25, color="orange", alpha=0.2, step="mid")
    ax4.set_ylabel("Wind (mph)")
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="upper left")

    fig.autofmt_xdate()
    fig.suptitle(f"{model_name} Meteogram – {location_label} – {run_label}", fontsize=14)


    # Panel 4: Wind (and gusts if available)
    ax_wind = axes[3]
    ax_wind.plot(times, df["wind10m_mph"], color="tab:green", label="Wind (mph)")
    if "gust_mph" in df.columns and df["gust_mph"].notna().any():
        ax_wind.plot(times, df["gust_mph"], color="tab:olive", linestyle="--", label="Gust (mph)")
    ax_wind.fill_between(times, 25, df["wind10m_mph"], where=df["wind10m_mph"] >= 25, color="orange", alpha=0.2, step="mid")
    ax_wind.set_ylabel("Wind (mph)")
    ax_wind.grid(True, alpha=0.3)
    ax_wind.legend(loc="upper left")

    # Optional Panel 5: Mean Sea Level Pressure
    if has_mslp:
        ax_p = axes[4]
        ax_p.plot(times, df["mslp_hpa"], color="tab:brown", label="MSLP (hPa)")
        ax_p.set_ylabel("MSLP (hPa)")
        ax_p.grid(True, alpha=0.3)
        ax_p.legend(loc="upper left")

    fig.autofmt_xdate()
    fig.suptitle(f"{model_name} Meteogram – {location_label} – {run_label}", fontsize=14)

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
    parser = argparse.ArgumentParser(description="GFS meteogram for a point location")
    parser.add_argument("--date", help="Run date in YYYYMMDD (UTC). Default: today (UTC, adjusted if needed).")
    parser.add_argument("--cycle", type=int, choices=[0, 6, 12, 18], help="GFS cycle hour (0/6/12/18 UTC). Default: most recent.")
    parser.add_argument("--hours", default="0-72:3", help="Forecast hours range, e.g., 0-120 or 0-72:6")
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT, help="Latitude for point forecast")
    parser.add_argument("--lon", type=float, default=DEFAULT_LON, help="Longitude for point forecast")
    parser.add_argument("--location-label", default="KCLE", help="Label for plots/output")
    parser.add_argument("--outdir", default="gfs_meteogram_output", help="Directory to store GRIB files and plot")
    parser.add_argument("--no-cache", action="store_true", help="Force re-download of GRIB files")
    parser.add_argument("--write-csv", action="store_true", help="Write the derived dataframe to CSV alongside the plot")

    args = parser.parse_args()
    now_utc = dt.datetime.utcnow()

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

    work_dir = Path(args.outdir) / f"gfs_{run_date.strftime('%Y%m%d')}_t{cycle:02d}z"
    plot_path = work_dir / f"meteogram_{args.location_label}_{run_date.strftime('%Y%m%d')}_t{cycle:02d}z.png"

    forecast = GFSPointForecast(run_date, cycle, args.lat, args.lon, hours, work_dir)
    forecast.fetch(no_cache=args.no_cache)
    df = forecast.to_dataframe()

    run_label = f"{run_date.strftime('%Y-%m-%d')} {cycle:02d}Z"
    plot_meteogram({forecast.model_name: df}, args.location_label, run_label, plot_path)

    if args.write_csv:
        csv_path = plot_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        print(f"[csv] Saved DataFrame to {csv_path}")


if __name__ == "__main__":
    main()
