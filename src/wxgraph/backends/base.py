"""Core interfaces for download and extraction backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence, Mapping


class BackendError(Exception):
    """Raised when a backend cannot satisfy a request."""


class FetchBackend(ABC):
    """Abstract base class for point-fetch backends."""

    @abstractmethod
    def download(
        self,
        model: str,
        fhours: Sequence[int],
        bbox: Mapping[str, float],
        outdir: Path,
        *,
        no_cache: bool,
    ) -> None:
        """Ensure the raw data files exist for the requested forecast hours."""

    @abstractmethod
    def extract_point(self, path: Path, lat: float, lon: float) -> dict[str, object]:
        """Extract a normalized point record from a raw dataset."""
