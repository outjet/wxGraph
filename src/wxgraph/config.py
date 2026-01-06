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
DEFAULT_SNOW_RATIO = os.environ.get("WXGRAPH_SNOW_RATIO", "10to1").strip().lower()
DEFAULT_BLEND_PERIODS = int(os.environ.get("WXGRAPH_BLEND_PERIODS", "0"))


def _parse_forecast_hours(value: str) -> Sequence[int]:
    hours: list[int] = []
    for fragment in value.split(","):
        fragment = fragment.strip()
        if not fragment:
            continue
        if ":" in fragment and "-" in fragment:
            range_part, step_part = fragment.split(":")
            start, end = [int(x) for x in range_part.split("-", 1)]
            step = int(step_part)
            hours.extend(range(start, end + 1, step))
        elif "-" in fragment:
            start, end = [int(x) for x in fragment.split("-", 1)]
            hours.extend(range(start, end + 1))
        else:
            hours.append(int(fragment))
    return sorted(set(hours))


def get_default_fhours() -> tuple[int, ...]:
    """Return forecast hours from WXGRAPH_FHOURS or the standard 3-hour grid."""

    value = os.environ.get("WXGRAPH_FHOURS")
    if not value:
        return tuple(range(0, 169, 3))
    parsed = _parse_forecast_hours(value)
    return tuple(parsed) if parsed else tuple(range(0, 169, 3))


DEFAULT_FHOURS = get_default_fhours()


def get_model_fhours() -> dict[str, tuple[int, ...]]:
    """Return per-model forecast hours from WXGRAPH_MODEL_FHOURS."""

    raw = os.environ.get("WXGRAPH_MODEL_FHOURS", "").strip()
    if not raw:
        return {}
    results: dict[str, tuple[int, ...]] = {}
    for entry in raw.split(";"):
        entry = entry.strip()
        if not entry or "=" not in entry:
            continue
        model, hours = entry.split("=", 1)
        model = model.strip().lower()
        parsed = _parse_forecast_hours(hours)
        if model and parsed:
            results[model] = tuple(parsed)
    return results


def get_herbie_models() -> set[str]:
    """Return models that should use Herbie downloads."""

    raw = os.environ.get("WXGRAPH_HERBIE_MODELS", "").strip()
    if not raw:
        return set()
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


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
