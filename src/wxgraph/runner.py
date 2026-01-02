"""Orchestrate downloads, pipelines, and exports for wxGraph."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path
import sys
import json
from typing import Callable, Mapping
import logging

import pandas as pd

from wxgraph.backends.base import FetchBackend
from wxgraph.backends.nomads_backend import NomadsBackend
from wxgraph.config import (
    DEFAULT_FHOURS,
    DEFAULT_LAT,
    DEFAULT_LOCATION_LABEL,
    DEFAULT_LON,
    get_blend_periods,
    get_models as get_default_models,
    get_output_dir,
    get_work_dir,
)
from wxgraph.models import AIGFS, GEFS, GFS, HGEFS, HRRR, NAM, RAP
from wxgraph.pipeline import add_dewpoint, add_precip_periods, add_wetbulb, merge_models, normalize
from wxgraph.pipeline.icing import add_icing_fields

MODEL_CLASSES: dict[str, type[object]] = {
    "gfs": GFS,
    "nam": NAM,
    "hrrr": HRRR,
    "rap": RAP,
    "gefs": GEFS,
    "aigfs": AIGFS,
    "hgefs": HGEFS,
}

CACHE_DIRNAME = "model_cache"
LOGGER = logging.getLogger("wxgraph.runner")

Plotter = Callable[[dict[str, pd.DataFrame], str, str, Path], None]


def _default_plotter(
    model_dfs: dict[str, pd.DataFrame],
    location_label: str,
    run_label: str,
    out_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from gfs_meteogram_kcle import plot_meteogram as _plot  # noqa: WPS433

    _plot(model_dfs, location_label, run_label, out_path)


class MeteogramRunner:
    """Execute a full fetch → pipeline → export workflow."""

    def __init__(
        self,
        *,
        models: Sequence[str] | None = None,
        lat: float = DEFAULT_LAT,
        lon: float = DEFAULT_LON,
        run_date: date,
        cycle: int,
        fhours: Sequence[int] | None = None,
        work_dir: Path | str | None = None,
        output_dir: Path | str | None = None,
        location_label: str = DEFAULT_LOCATION_LABEL,
        backend: FetchBackend | None = None,
        plotter: Plotter | None = None,
        no_cache: bool = False,
    ) -> None:
        self.models = [
            m.lower()
            for m in (
                models
                or get_default_models(MODEL_CLASSES.keys())
            )
        ]
        self.lat = lat
        self.lon = lon
        self.run_date = run_date
        self.cycle = cycle
        self.fhours = tuple(fhours or DEFAULT_FHOURS)
        self.work_dir = Path(work_dir or get_work_dir())
        self.output_dir = Path(output_dir or get_output_dir())
        self.location_label = location_label
        self.backend = backend or NomadsBackend()
        self.no_cache = no_cache
        self.plotter = plotter or _default_plotter

    @property
    def run_label(self) -> str:
        """Text label describing the run."""
        return f"{self.run_date:%Y%m%d} t{self.cycle:02d}z"

    def run(self) -> Mapping[str, Path]:
        """Fetch, process, and export model data."""

        model_dfs: dict[str, pd.DataFrame] = {}
        status = {
            "generated_at": datetime.utcnow().isoformat(),
            "run_label": self.run_label,
            "run_date": self.run_date.strftime("%Y%m%d"),
            "cycle": self.cycle,
            "models": {},
        }
        for model_key in self.models:
            model_cls = MODEL_CLASSES.get(model_key)
            if model_cls is None:
                raise ValueError(f"Unsupported model key: {model_key}")
            model = model_cls(
                run_date=self.run_date,
                cycle=self.cycle,
                lat=self.lat,
                lon=self.lon,
                fhours=self.fhours,
                work_dir=self.work_dir / model_key,
            )
            missing_fhours: list[int] = []
            try:
                model.fetch(self.backend, no_cache=self.no_cache)
                df = model.to_dataframe(self.backend)
                missing_fhours = df.attrs.get("missing_fhours", [])
            except Exception as exc:
                LOGGER.warning("Model %s failed to fetch: %s", model_key, exc)
                status["models"][model_key] = {
                    "ok": False,
                    "error": str(exc),
                    "missing_fhours": list(self.fhours),
                }
                df = pd.DataFrame()
                missing_fhours = list(self.fhours)
            processed = self._prepare_model_dataframe(df, model.model_name)
            combined = self._merge_with_cache(model_key, processed, missing_fhours)
            if combined is not None and not combined.empty:
                model_dfs[model.model_name] = combined
                status["models"][model_key] = {
                    "ok": True,
                    "missing_fhours": list(missing_fhours),
                    "total_fhours": len(self.fhours),
                    "used_fallback": bool(missing_fhours),
                }
            elif model_key not in status["models"]:
                status["models"][model_key] = {
                    "ok": False,
                    "missing_fhours": list(self.fhours),
                    "total_fhours": len(self.fhours),
                    "used_fallback": False,
                }

        if not model_dfs:
            self._write_status(status, success=False, detail="No model data was produced.")
            raise RuntimeError("No model data was produced.")

        merged = merge_models(model_dfs)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.output_dir / "meteogram_latest.json"
        merged.to_json(json_path, orient="records", date_format="iso")
        png_path = self.output_dir / "meteogram_latest.png"
        self.plotter(model_dfs, self.location_label, self.run_label, png_path)
        self._write_status(status, success=True, detail=None)
        return {"json": json_path, "png": png_path, "merged": merged}

    def _prepare_model_dataframe(self, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if df.empty:
            return df
        return self._process_model_dataframe(df, prefix)

    def _write_status(self, status: dict, *, success: bool, detail: str | None) -> None:
        status["success"] = success
        if detail:
            status["detail"] = detail
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "meteogram_status.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(status, handle, indent=2, sort_keys=True)

    def _cache_dir(self) -> Path:
        return self.output_dir / CACHE_DIRNAME

    def _cache_path(self, model_key: str, label: str) -> Path:
        return self._cache_dir() / f"{model_key}_{label}.json"

    def _load_cached(self, model_key: str, label: str) -> pd.DataFrame | None:
        path = self._cache_path(model_key, label)
        if not path.exists():
            return None
        df = pd.read_json(path, orient="records")
        return df

    def _save_cached(self, model_key: str, label: str, df: pd.DataFrame) -> None:
        cache_dir = self._cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(model_key, label)
        df.to_json(path, orient="records", date_format="iso")

    def _merge_with_cache(
        self,
        model_key: str,
        latest: pd.DataFrame,
        missing_fhours: Sequence[int],
    ) -> pd.DataFrame | None:
        cached_complete = self._load_cached(model_key, "complete")
        cached_latest = self._load_cached(model_key, "latest")
        fallback = cached_complete if cached_complete is not None else cached_latest

        combined = latest
        if missing_fhours:
            if fallback is not None and not fallback.empty:
                blended = self._blend_runs(latest, fallback, get_blend_periods())
                combined = self._prefer_latest(blended, fallback)
            elif latest.empty:
                combined = latest
        if not latest.empty:
            self._save_cached(model_key, "latest", latest)
            if not missing_fhours:
                self._save_cached(model_key, "complete", latest)
        return combined

    def _prefer_latest(self, latest: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
        if latest.empty:
            return fallback
        if fallback.empty:
            return latest
        combined = pd.concat([latest, fallback], ignore_index=True, sort=False)
        key = None
        if "valid_time" in combined:
            key = "valid_time"
            combined[key] = pd.to_datetime(combined[key])
        elif "forecast_hour" in combined:
            key = "forecast_hour"
        if key is not None:
            combined = combined.drop_duplicates(subset=[key], keep="first")
            if key == "valid_time":
                combined = combined.sort_values(key)
        return combined.reset_index(drop=True)

    def _blend_runs(
        self,
        latest: pd.DataFrame,
        fallback: pd.DataFrame,
        periods: int,
    ) -> pd.DataFrame:
        if latest.empty or fallback.empty or periods <= 0:
            return latest
        time_key = "valid_time" if "valid_time" in latest else "forecast_hour"
        latest = latest.copy()
        if time_key == "valid_time":
            latest[time_key] = pd.to_datetime(latest[time_key])
            fallback = fallback.copy()
            fallback[time_key] = pd.to_datetime(fallback[time_key])
        seam_times = (
            latest[time_key]
            .dropna()
            .sort_values()
            .unique()
            .tolist()[:periods]
        )
        if not seam_times:
            return latest
        numeric_cols = latest.select_dtypes(include="number").columns.tolist()
        skip_cols = {"forecast_hour", "f_hour"}
        numeric_cols = [col for col in numeric_cols if col not in skip_cols]
        fallback_numeric = fallback.select_dtypes(include="number").columns.tolist()
        numeric_cols = [col for col in numeric_cols if col in fallback_numeric]
        if not numeric_cols:
            return latest
        fallback_indexed = fallback.set_index(time_key)
        latest_indexed = latest.set_index(time_key)
        for idx, seam_time in enumerate(seam_times):
            if seam_time not in fallback_indexed.index or seam_time not in latest_indexed.index:
                continue
            weight_new = (idx + 1) / max(1, len(seam_times))
            weight_old = 1.0 - weight_new
            latest_vals = latest_indexed.loc[seam_time, numeric_cols]
            fallback_vals = fallback_indexed.loc[seam_time, numeric_cols]
            blended = latest_vals * weight_new + fallback_vals * weight_old
            latest_indexed.loc[seam_time, numeric_cols] = blended
        latest = latest_indexed.reset_index()
        return latest

    def _process_model_dataframe(self, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        df = normalize(df)
        df = add_dewpoint(df)
        df = add_wetbulb(df)
        df = add_precip_periods(df)
        dewpoint = df.get("dewpoint_c")
        if dewpoint is None:
            dewpoint = pd.Series(0.0, index=df.index, dtype=float)
        df["dewpoint_k"] = dewpoint + 273.15
        qpf_in = df.get("qpf_in_raw")
        if qpf_in is None:
            qpf_in = pd.Series(0.0, index=df.index, dtype=float)
        df["qpf_mm"] = qpf_in * 25.4
        snow_acc = df.get("snow_acc_in")
        if snow_acc is None:
            snow_acc = pd.Series(0.0, index=df.index, dtype=float)
        df["snow_mm"] = snow_acc * 25.4
        df["model"] = prefix
        return add_icing_fields(
            df,
            prefix=prefix,
            time_col="valid_time",
            temp_k_col="temp_k",
            rh_pct_col="rh_percent",
            dp_k_col="dewpoint_k",
            qpf_accum_mm_col="qpf_mm",
            snow_accum_mm_col="snow_mm",
            gust_mph_col="gust_mph",
            wind_mph_col="wind10m_mph",
            cloud_pct_col="cloud_pct",
            latitude=self.lat,
        )
