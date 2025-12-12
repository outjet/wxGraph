"""NOMADS-backed implementation of :class:`FetchBackend`."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Mapping

from wxgraph.backends.base import BackendError, FetchBackend


class NomadsBackend(FetchBackend):
    """Fetch backend that relies on the NOMADS grib-filter API."""

    def download(
        self,
        model: str,
        fhours: Sequence[int],
        bbox: Mapping[str, float],
        outdir: Path,
        *,
        no_cache: bool,
    ) -> None:
        """
        Download or reuse per-hour GRIB2 files from the NOMADS filter service.
        """

        outdir.mkdir(parents=True, exist_ok=True)
        for fh in fhours:
            url = _build_nomads_url(model, fh, bbox)
            out_path = outdir / f"{model}_{fh}.grib2"
            _download_grib(url, out_path, no_cache=no_cache)

    def extract_point(self, path: Path, lat: float, lon: float) -> dict[str, object]:
        """
        Extract a normalized point record from a NOMADS GRIB dataset.
        """

        raise BackendError("Nomads point extraction is not implemented yet")


def _build_nomads_url(model: str, fh: int, bbox: Mapping[str, float]) -> str:
    """
    Construct a placeholder NOMADS URL for a model and forecast hour.
    """

    return f"https://nomads.example.com/{model}/f{fh:03d}"


def _download_grib(url: str, path: Path, *, no_cache: bool) -> None:
    """
    Download a GRIB2 payload when caching is disabled.
    """

    if path.exists() and not no_cache:
        return
    raise BackendError(f"NOMADS download for {url} is not implemented")
