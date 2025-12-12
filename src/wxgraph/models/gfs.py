"""GFS point-forecast adapter."""

from __future__ import annotations

from wxgraph.models.base import BasePointModel


class GFS(BasePointModel):
    """Adapter for GFS 0.25° NOMADS data."""

    model_key = "gfs"
    file_prefix = "gfs"
