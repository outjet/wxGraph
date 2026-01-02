"""Example runner that wires together the new wxGraph modules."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from wxgraph.backends.nomads_backend import NomadsBackend
from wxgraph.models.gfs import GFS
from wxgraph.pipeline import normalize


def run_example() -> None:
    """
    Download a small GFS sample and print the normalized summary.
    """

    work_dir = Path("data/gfs_example")
    backend = NomadsBackend()
    model = GFS(
        run_date=date(2024, 1, 1),
        cycle=0,
        lat=41.48,
        lon=-81.81,
        fhours=[0, 3, 6],
        work_dir=work_dir,
    )
    model.fetch(backend, no_cache=True)
    df = model.to_dataframe(backend)
    normalized = normalize(df)
    print(normalized.head())


if __name__ == "__main__":
    run_example()
