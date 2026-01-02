"""Abstract point-model definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path
from typing import Sequence, Mapping
import logging

import pandas as pd

from wxgraph.backends.base import FetchBackend


DEFAULT_POINT_BBOX: Mapping[str, float] = {
    "left": -84.0,
    "right": -79.0,
    "top": 43.0,
    "bottom": 40.0,
}


class PointModel(ABC):
    """Defines the high-level point-forecast adapter interface."""

    @abstractmethod
    def fetch(self, backend: FetchBackend, *, no_cache: bool = False) -> None:
        """Download all data for this model via the provided backend."""

    @abstractmethod
    def to_dataframe(self, backend: FetchBackend) -> pd.DataFrame:
        """Return a normalized DataFrame of extracted point records."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the human-readable model name."""


class BasePointModel(PointModel):
    """Shared implementation for simple point-model adapters."""

    model_key: str
    file_prefix: str
    default_bbox: Mapping[str, float] = DEFAULT_POINT_BBOX

    def __init__(
        self,
        run_date: date,
        cycle: int,
        lat: float,
        lon: float,
        fhours: Sequence[int],
        work_dir: Path,
        *,
        bbox: Mapping[str, float] | None = None,
    ) -> None:
        """
        Store the core request parameters.
        """

        self.run_date = run_date
        self.cycle = cycle
        self.lat = lat
        self.lon = lon
        self.fhours = tuple(fhours)
        self.work_dir = work_dir
        self.bbox = dict(bbox or self.default_bbox)

    @property
    def model_name(self) -> str:
        """Default model name implementation based on the backend key."""

        return self.model_key.upper()

    def fetch(self, backend: FetchBackend, *, no_cache: bool = False) -> None:
        """
        Delegate download operations to the backend.
        """

        metadata = {"run_date": self.run_date, "cycle": self.cycle, "bbox": self.bbox}
        backend.download(
            self.model_key,
            self.fhours,
            self.bbox,
            self.work_dir,
            no_cache=no_cache,
            metadata=metadata,
        )

    def to_dataframe(self, backend: FetchBackend) -> pd.DataFrame:
        """
        Collect point records for each forecast hour.
        """

        records: list[dict[str, object]] = []
        missing: list[int] = []
        logger = logging.getLogger("wxgraph.models")
        for fh in self.fhours:
            path = self._record_path(fh)
            try:
                record = backend.extract_point(path, self.lat, self.lon, fh)
            except (FileNotFoundError, EOFError, OSError, ValueError) as exc:
                missing.append(fh)
                if path.exists():
                    path.unlink(missing_ok=True)
                logger.warning("Missing or corrupt %s fhour %s: %s", self.model_key, fh, exc)
                continue
            record["model"] = self.model_name
            record["forecast_hour"] = fh
            records.append(record)
        df = pd.DataFrame(records)
        df.attrs["missing_fhours"] = missing
        return df

    def _record_path(self, fh: int) -> Path:
        """Build the expected GRIB2 file path for a forecast hour."""

        return self.work_dir / f"{self.file_prefix}_{fh}.grib2"
