"""Backend that routes downloads by model to Herbie or NOMADS."""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

from wxgraph.backends.base import BackendError, FetchBackend
from wxgraph.backends.cfgrib_helpers import extract_surface_record
from wxgraph.backends.herbie_backend import HerbieBackend
from wxgraph.backends.nomads_backend import NomadsBackend

LOGGER = logging.getLogger("wxgraph.backends")


class RoutedBackend(FetchBackend):
    """Route downloads to Herbie for select models and fall back to NOMADS."""

    def __init__(self, herbie_models: Sequence[str] | None = None) -> None:
        self.herbie_models = {model.lower() for model in (herbie_models or [])}
        self.herbie = HerbieBackend()
        self.nomads = NomadsBackend()

    def download(
        self,
        model: str,
        fhours: Sequence[int],
        bbox: Mapping[str, float],
        outdir,
        *,
        no_cache: bool,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        use_herbie = model.lower() in self.herbie_models and self.herbie._is_available()
        if use_herbie:
            try:
                self.herbie.download(
                    model,
                    fhours,
                    bbox,
                    outdir,
                    no_cache=no_cache,
                    metadata=metadata,
                )
                return
            except Exception as exc:
                LOGGER.warning("Herbie failed for %s, falling back to NOMADS: %s", model, exc)
        self.nomads.download(
            model,
            fhours,
            bbox,
            outdir,
            no_cache=no_cache,
            metadata=metadata,
        )

    def extract_point(
        self,
        path,
        lat: float,
        lon: float,
        fh: int | None = None,
    ) -> dict[str, object]:
        return extract_surface_record(path, lat, lon, fh)
