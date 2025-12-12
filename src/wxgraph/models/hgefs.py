"""HGEFS point-forecast adapter."""

from __future__ import annotations

from wxgraph.models.base import BasePointModel


class HGEFS(BasePointModel):
    """Adapter for HGEFS ensemble statistics."""

    model_key = "hgefs"
    file_prefix = "hgefs"
