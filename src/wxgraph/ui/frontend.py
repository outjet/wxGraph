"""Simple frontend helpers for wxGraph."""

from __future__ import annotations

from typing import Mapping

import pandas as pd


def summarize_point_data(df: pd.DataFrame) -> Mapping[str, object]:
    """
    Prepare a compact summary of a point forecast for JSON serialization.
    """

    return {
        "records": len(df),
        "model_breakdown": df["model"].unique().tolist() if "model" in df else [],
    }
