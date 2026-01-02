"""Herbie-backed implementation of :class:`FetchBackend`."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from datetime import datetime, date, time
from pathlib import Path
from typing import Sequence

import shutil

try:
    from herbie import Herbie
except ImportError:  # pragma: no cover - optional dependency
    Herbie = None

from wxgraph.backends.base import BackendError, FetchBackend
from wxgraph.backends.cfgrib_helpers import extract_surface_record


class HerbieBackend(FetchBackend):
    """Fetch backend that delegates to Herbie for downloads and extraction."""

    @staticmethod
    def _is_available() -> bool:
        """Return True if Herbie has been installed."""

        return Herbie is not None

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
        Use Herbie to request regional data if the library is available.
        """

        if Herbie is None:
            raise BackendError("Herbie is not installed.")

        run_date, cycle = self._metadata_from_context(metadata)
        product = str(metadata.get("product", "sfc")) if metadata else "sfc"
        outdir.mkdir(parents=True, exist_ok=True)

        for fh in fhours:
            run_dt = datetime.combine(run_date, time(cycle))
            herbie_obj = Herbie(
                run_dt,
                model=model,
                product=product,
                fxx=fh,
                save_dir=outdir,
                overwrite=no_cache,
                verbose=False,
            )
            grib_path = herbie_obj.download(errors="raise", overwrite=no_cache)
            dest = outdir / f"{model}_{fh}.grib2"
            shutil.move(str(grib_path), dest)

    def extract_point(
        self,
        path: Path,
        lat: float,
        lon: float,
        fh: int | None = None,
    ) -> dict[str, object]:
        """Use Herbie.sample to extract the nearest grid point."""

        if not path.exists():
            raise FileNotFoundError(f"Missing GRIB file: {path}")
        return extract_surface_record(path, lat, lon, fh)

    def _metadata_from_context(self, metadata: MappingABC[str, object] | None) -> tuple[date, int]:
        if metadata is None:
            raise BackendError("Missing metadata for Herbie download")
        run_date = metadata.get("run_date")
        cycle = metadata.get("cycle")
        if not isinstance(run_date, date) or not isinstance(cycle, int):
            raise BackendError("run_date and cycle metadata are required for Herbie downloads")
        return run_date, cycle
