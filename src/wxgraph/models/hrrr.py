"""HRRR point-forecast adapter."""

from __future__ import annotations

from wxgraph.models.base import BasePointModel


class HRRR(BasePointModel):
    """Adapter for HRRR surface data."""

    model_key = "hrrr"
    file_prefix = "hrrr"
