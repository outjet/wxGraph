"""GEFS point-forecast adapter."""

from __future__ import annotations

from wxgraph.models.base import BasePointModel


class GEFS(BasePointModel):
    """Adapter for GEFS ensemble means."""

    model_key = "gefs"
    file_prefix = "gefs"
