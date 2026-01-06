"""NOMADS-backed implementation of :class:`FetchBackend`."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from datetime import date
from pathlib import Path
from typing import Sequence

import datetime as dt
import logging
import requests

from wxgraph.backends.base import BackendError, FetchBackend
from wxgraph.backends.cfgrib_helpers import extract_surface_record
from wxgraph.backends.nomads_urls import build_nomads_urls


class NomadsBackend(FetchBackend):
    """Fetch backend that relies on NOMADS URLs and cfgrib extraction."""

    def download(
        self,
        model: str,
        fhours: Sequence[int],
        bbox: MappingABC[str, float],
        outdir: Path,
        *,
        no_cache: bool,
        metadata: MappingABC[str, object] | None = None,
    ) -> None:
        """
        Download or reuse per-hour GRIB2 files from NOMADS.
        """

        run_date, cycle = self._metadata_from_context(metadata)
        outdir.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("wxgraph.nomads")

        missing: list[int] = []
        for fh in fhours:
            url_candidates = build_nomads_urls(model, run_date, cycle, fh, bbox)
            dest = outdir / f"{model}_{fh}.grib2"
            if not self._download_from_candidates(url_candidates, dest, no_cache=no_cache):
                missing.append(fh)
        if missing:
            logger.warning("Missing %s fhours for %s: %s", len(missing), model, missing)

    def extract_point(
        self,
        path: Path,
        lat: float,
        lon: float,
        fh: int | None = None,
    ) -> dict[str, object]:
        """Extract the point record using cfgrib helpers."""

        if not path.exists():
            raise FileNotFoundError(f"Missing GRIB file: {path}")
        return extract_surface_record(path, lat, lon, fh)

    def _metadata_from_context(self, metadata: MappingABC[str, object] | None) -> tuple[date, int]:
        if metadata is None:
            raise BackendError("Missing metadata for NOMADS download")
        run_date = metadata.get("run_date")
        cycle = metadata.get("cycle")
        if not isinstance(run_date, date) or not isinstance(cycle, int):
            raise BackendError("run_date and cycle metadata are required for NOMADS downloads")
        return run_date, cycle

    def _download_from_candidates(self, urls: list[str], path: Path, *, no_cache: bool) -> bool:
        last_exc: requests.HTTPError | None = None
        for url in urls:
            try:
                _download_grib(url, path, no_cache=no_cache)
                return True
            except requests.HTTPError as exc:
                last_exc = exc
                if _is_missing_http_error(exc):
                    continue
                raise
        if last_exc is not None:
            if path.exists():
                return True
            return False
        return False


def _download_grib(url: str, path: Path, *, no_cache: bool) -> None:
    """
    Download a GRIB2 file, honor caching, and write to ``path``.
    """

    if path.exists() and not no_cache:
        age = dt.datetime.now() - dt.datetime.fromtimestamp(path.stat().st_mtime)
        if age < dt.timedelta(hours=1):
            return

    response = requests.get(url, timeout=120)
    response.raise_for_status()
    if not response.content.startswith(b"GRIB"):
        response.status_code = 404
        raise requests.HTTPError("Invalid GRIB payload", response=response)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_bytes(response.content)
    tmp_path.replace(path)


def _is_missing_http_error(exc: requests.HTTPError) -> bool:
    if exc.response is None:
        return False
    return exc.response.status_code in {403, 404}
