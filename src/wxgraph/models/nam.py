"""NAM point-forecast adapter."""

from __future__ import annotations

from wxgraph.models.base import BasePointModel


class NAM(BasePointModel):
    """Adapter for NAM surface data."""

    model_key = "nam"
    file_prefix = "nam"
