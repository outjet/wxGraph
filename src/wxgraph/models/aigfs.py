"""AIGFS point-forecast adapter."""

from __future__ import annotations

from wxgraph.models.base import BasePointModel


class AIGFS(BasePointModel):
    """Adapter for AIGFS deterministic surface forecasts."""

    model_key = "aigfs"
    file_prefix = "aigfs"
