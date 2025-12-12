"""Point-forecast model adapters."""

from __future__ import annotations

from .aigfs import AIGFS
from .gefs import GEFS
from .gfs import GFS
from .hgefs import HGEFS
from .hrrr import HRRR
from .nam import NAM
from .rap import RAP

__all__ = ["AIGFS", "GEFS", "GFS", "HGEFS", "HRRR", "NAM", "RAP"]
