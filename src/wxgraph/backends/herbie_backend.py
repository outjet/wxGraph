"""Herbie-backed implementation of :class:`FetchBackend`."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Mapping

from wxgraph.backends.base import BackendError, FetchBackend

try:
    from herbie import Herbie  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency
    Herbie = None


class HerbieBackend(FetchBackend):
    """Fetch backend that delegates to Herbie for downloads and extraction."""

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
        Use Herbie to request regional data if the library is available.
        """

        if Herbie is None:
            raise BackendError("Herbie is not installed")
        for fh in fhours:
            _queue_herbie_request(model, fh, bbox, outdir, no_cache=no_cache)

    def extract_point(self, path: Path, lat: float, lon: float) -> dict[str, object]:
        """
        Use Herbie's xarray wrapper to sample the nearest grid point.
        """

        raise BackendError("Herbie extraction is not implemented yet")


def _queue_herbie_request(
    model: str,
    fh: int,
    bbox: Mapping[str, float],
    outdir: Path,
    *,
    no_cache: bool,
) -> None:
    """
    Prepare a Herbie job for the requested forecast hour.
    """

    if outdir.exists() and not no_cache:
        return
    raise BackendError(f"Herbie download for {model} f{fh:03d} is not implemented")
