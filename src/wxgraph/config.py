"""Shared configuration helpers for wxGraph."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"


def _resolve_path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if not value:
        return default
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    return candidate


def get_output_dir() -> Path:
    """Return where meteogram artifacts should be written."""

    return _resolve_path_from_env("WXGRAPH_OUTPUT_DIR", REPO_ROOT / "gfs_meteogram_output")


def get_work_dir() -> Path:
    """Return the default working directory for downloads."""

    return _resolve_path_from_env("WXGRAPH_WORK_DIR", REPO_ROOT / "data")


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists."""

    path.mkdir(parents=True, exist_ok=True)
    return path


DEFAULT_LAT = float(os.environ.get("WXGRAPH_LAT", "41.48"))
DEFAULT_LON = float(os.environ.get("WXGRAPH_LON", "-81.81"))
DEFAULT_LOCATION_LABEL = os.environ.get("WXGRAPH_SITE", "Point")
DEFAULT_FHOURS = tuple(range(0, 169, 3))
DEFAULT_SNOW_RATIO = os.environ.get("WXGRAPH_SNOW_RATIO", "10to1").strip().lower()
DEFAULT_BLEND_PERIODS = int(os.environ.get("WXGRAPH_BLEND_PERIODS", "0"))


def get_models(defaults: Iterable[str]) -> list[str]:
    """Return the ordered list of models to use."""

    override = os.environ.get("WXGRAPH_MODELS")
    if not override:
        return list(defaults)
    return [item.strip().lower() for item in override.split(",") if item.strip()]


def get_snow_ratio_method() -> str:
    """Return the configured snow ratio method."""

    return DEFAULT_SNOW_RATIO


def get_blend_periods() -> int:
    """Return how many forecast steps to blend across run seams."""

    return max(0, DEFAULT_BLEND_PERIODS)
