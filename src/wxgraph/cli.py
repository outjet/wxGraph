"""Command-line entry point for wxGraph."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Sequence

import click

from wxgraph.backends.herbie_backend import HerbieBackend
from wxgraph.backends.nomads_backend import NomadsBackend
from wxgraph.config import get_output_dir, get_work_dir
from wxgraph.runner import MeteogramRunner


def _parse_model_list(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def _parse_forecast_hours(value: str) -> Sequence[int]:
    hours: list[int] = []
    for fragment in value.split(","):
        fragment = fragment.strip()
        if not fragment:
            continue
        if ":" in fragment and "-" in fragment:
            range_part, step_part = fragment.split(":")
            start, end = [int(x) for x in range_part.split("-", 1)]
            step = int(step_part)
            hours.extend(range(start, end + 1, step))
        elif "-" in fragment:
            start, end = [int(x) for x in fragment.split("-", 1)]
            hours.extend(range(start, end + 1))
        else:
            hours.append(int(fragment))
    return sorted(set(hours))


@click.command()
@click.option("--lat", type=float, default=41.48, help="Latitude for the point fetch.")
@click.option("--lon", type=float, default=-81.81, help="Longitude for the point fetch.")
@click.option("--run-date", default=date.today().strftime("%Y%m%d"), help="YYYYMMDD model run date.")
@click.option("--cycle", type=click.Choice(["0", "6", "12", "18"]), default="0", help="Model cycle hour.")
@click.option("--models", default="gfs", help="Comma-separated models: gfs, nam, hrrr, rap, gefs, aigfs, hgefs.")
@click.option("--fhours", default="0-12:3", help="Forecast hours (e.g., 0-12:3 or 0,3,6).")
@click.option("--work-dir", type=click.Path(path_type=Path), default=get_work_dir(), help="Working directory.")
@click.option("--output-dir", type=click.Path(path_type=Path), default=get_output_dir(), help="Export directory.")
@click.option("--location-label", default="Point", help="Label used in the exported summary and plot.")
@click.option("--backend", type=click.Choice(["nomads", "herbie"]), default="nomads", help="Fetch backend.")
@click.option("--no-cache", is_flag=True, help="Always download fresh GRIB2 files.")
def main(
    lat: float,
    lon: float,
    run_date: str,
    cycle: str,
    models: str,
    fhours: str,
    work_dir: Path,
    output_dir: Path,
    location_label: str,
    backend: str,
    no_cache: bool,
) -> None:
    """Fetch models, run the pipelines, and export PNG/JSON for the web app."""

    parsed_date = datetime.strptime(run_date, "%Y%m%d").date()
    model_list = _parse_model_list(models)
    forecast_hours = _parse_forecast_hours(fhours)
    backend_impl = HerbieBackend() if backend == "herbie" else NomadsBackend()

    runner = MeteogramRunner(
        models=model_list,
        lat=lat,
        lon=lon,
        run_date=parsed_date,
        cycle=int(cycle),
        fhours=forecast_hours,
        work_dir=work_dir,
        output_dir=output_dir,
        location_label=location_label,
        backend=backend_impl,
        no_cache=no_cache,
    )

    result = runner.run()
    click.echo(f"Output JSON written to {result['json']}")
    click.echo(f"Meteogram PNG written to {result['png']}")
