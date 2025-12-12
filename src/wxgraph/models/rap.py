"""RAP point-forecast adapter."""

from __future__ import annotations

from wxgraph.models.base import BasePointModel


class RAP(BasePointModel):
    """Adapter for RAP surface data."""

    model_key = "rap"
    file_prefix = "rap"
