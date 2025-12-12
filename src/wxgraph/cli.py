"""Command-line entry point for wxGraph."""

from __future__ import annotations

from pathlib import Path

import click

from wxgraph.backends.nomads_backend import NomadsBackend
from wxgraph.models.gfs import GFS


@click.command()
@click.option("--lat", type=float, default=41.48, help="Latitude for the point fetch.")
@click.option("--lon", type=float, default=-81.81, help="Longitude for the point fetch.")
@click.option("--work-dir", type=click.Path(path_type=Path), default=Path("data"), help="Working directory.")
def main(lat: float, lon: float, work_dir: Path) -> None:
    """
    Simple CLI that downloads a minimal GFS sample and prints a summary.
    """

    backend = NomadsBackend()
    model = GFS(
        run_date="20240101",
        cycle=0,
        lat=lat,
        lon=lon,
        fhours=[0, 3, 6, 9],
        work_dir=work_dir,
    )
    model.fetch(backend, no_cache=True)
    df = model.to_dataframe(backend)
    click.echo(f"Fetched {len(df)} rows from {model.model_name}")
