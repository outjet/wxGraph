"""LCR alerting helpers backed by wxGraph outputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
import os
from typing import Any

import pandas as pd
import requests

from wxgraph.storage import read_gcs_json, write_gcs_json

LOGGER = logging.getLogger("wxgraph.alerts.lcr")


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


@dataclass(frozen=True)
class AlertConfig:
    enabled: bool
    model: str
    hours_ahead: int
    threshold: int
    cooldown_hours: int
    state_name: str
    location_label: str
    pushover_app_token: str | None
    pushover_user_key: str | None

    @classmethod
    def from_env(cls) -> "AlertConfig":
        return cls(
            enabled=_parse_bool(os.environ.get("WXGRAPH_LCR_ALERT_ENABLED")),
            model=os.environ.get("WXGRAPH_LCR_ALERT_MODEL", "HRRR"),
            hours_ahead=_parse_int(os.environ.get("WXGRAPH_LCR_ALERT_HOURS_AHEAD"), 12),
            threshold=_parse_int(os.environ.get("WXGRAPH_LCR_ALERT_THRESHOLD"), 5),
            cooldown_hours=_parse_int(os.environ.get("WXGRAPH_LCR_ALERT_COOLDOWN_HOURS"), 4),
            state_name=os.environ.get("WXGRAPH_LCR_ALERT_STATE_NAME", "lcr_alert_state.json"),
            location_label=os.environ.get("WXGRAPH_SITE", "Point"),
            pushover_app_token=os.environ.get("WXGRAPH_PUSHOVER_APP_TOKEN"),
            pushover_user_key=os.environ.get("WXGRAPH_PUSHOVER_USER_KEY"),
        )


def _select_window(
    df: pd.DataFrame,
    *,
    model: str,
    hours_ahead: int,
) -> tuple[pd.Series, datetime] | None:
    if df.empty:
        return None

    model_name = model.upper()
    lcr_col = f"{model_name}_lcr"
    if "source_model" in df:
        model_df = df[df["source_model"].str.upper() == model_name]
    else:
        model_df = df
    if model_df.empty:
        LOGGER.warning("LCR alert skipped: no rows for model %s", model_name)
        return None
    if lcr_col not in model_df.columns:
        LOGGER.warning("LCR alert skipped: missing column %s", lcr_col)
        return None

    times = pd.to_datetime(model_df["valid_time"], errors="coerce", utc=True)
    lcr_vals = pd.to_numeric(model_df[lcr_col], errors="coerce")
    valid_mask = times.notna() & lcr_vals.notna()
    if not valid_mask.any():
        LOGGER.warning("LCR alert skipped: no valid LCR/time values")
        return None

    times = times[valid_mask]
    lcr_vals = lcr_vals[valid_mask]
    start = times.min()
    end = start + pd.Timedelta(hours=hours_ahead)
    window_mask = (times >= start) & (times <= end)
    if not window_mask.any():
        LOGGER.warning("LCR alert skipped: no values in window")
        return None

    window_lcr = lcr_vals[window_mask]
    idx = window_lcr.idxmax()
    max_lcr = window_lcr.loc[idx]
    max_time = times.loc[idx]
    return pd.Series({"lcr": max_lcr, "valid_time": max_time}), start.to_pydatetime()


def _cooldown_elapsed(state: dict[str, Any], now: datetime, cooldown_hours: int) -> bool:
    last_alert = _parse_iso(state.get("last_alert_time_utc"))
    if last_alert is None:
        return True
    elapsed = now - last_alert
    return elapsed >= timedelta(hours=cooldown_hours)


def _send_pushover(
    token: str,
    user_key: str,
    *,
    title: str,
    message: str,
    priority: int = 0,
) -> None:
    response = requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": token,
            "user": user_key,
            "title": title,
            "message": message,
            "priority": str(priority),
        },
        timeout=10,
    )
    response.raise_for_status()


def maybe_send_lcr_alert(df: pd.DataFrame) -> None:
    config = AlertConfig.from_env()
    if not config.enabled:
        return

    selection = _select_window(df, model=config.model, hours_ahead=config.hours_ahead)
    if selection is None:
        return

    max_info, _ = selection
    max_lcr = int(max_info["lcr"])
    max_time = pd.to_datetime(max_info["valid_time"], utc=True).to_pydatetime()

    if max_lcr < config.threshold:
        LOGGER.info("LCR alert not triggered (max=%s, threshold=%s).", max_lcr, config.threshold)
        return

    if not config.pushover_app_token or not config.pushover_user_key:
        LOGGER.warning("LCR alert skipped: missing Pushover credentials.")
        return

    state = read_gcs_json(config.state_name) or {}
    now = datetime.now(timezone.utc)
    if not _cooldown_elapsed(state, now, config.cooldown_hours):
        LOGGER.info("LCR alert cooldown active.")
        return

    title = f"LCR alert: {config.location_label}"
    time_str = max_time.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    message = f"{config.model.upper()} max LCR {max_lcr} at {time_str}"

    try:
        _send_pushover(
            config.pushover_app_token,
            config.pushover_user_key,
            title=title,
            message=message,
            priority=0,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Pushover send failed: %s", exc)
        return

    state_update = {
        "last_alert_time_utc": now.isoformat(),
        "last_lcr": max_lcr,
        "last_valid_time_utc": max_time.isoformat(),
        "model": config.model.upper(),
    }
    write_gcs_json(config.state_name, state_update)
