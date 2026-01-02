import pandas as pd
import pytest

from wxgraph.pipeline.derive import add_dewpoint, add_precip_periods, add_wetbulb


def test_add_dewpoint_computes_basics():
    df = pd.DataFrame(
        {
            "temp_c": [0.0, 5.0],
            "rh_percent": [80.0, 30.0],
        }
    )
    result = add_dewpoint(df)
    assert result["dewpoint_c"].iloc[0] > result["dewpoint_c"].iloc[1]


def test_add_precip_periods_derives_increments_and_types():
    df = pd.DataFrame(
        {
            "valid_time": pd.date_range("2024-01-01", periods=3, freq="3h"),
            "temp_c": [-1.0, 1.0, 5.0],
            "qpf_in_raw": [0.0, 0.2, 0.5],
        }
    )
    result = add_precip_periods(df)
    assert result["qpf_in"].iloc[1] == pytest.approx(0.2)
    assert result["qpf_in"].iloc[2] == pytest.approx(0.3)
    assert result["precip_type"].tolist() == ["none", "mix", "rain"]
    assert result["snow_acc_in"].iloc[-1] >= 0.0


def test_add_wetbulb_creates_values():
    df = pd.DataFrame(
        {
            "temp_c": [0.0, 10.0],
            "rh_percent": [90.0, 50.0],
        }
    )
    result = add_wetbulb(df)
    assert "wetbulb_c" in result
    assert result["wetbulb_c"].notna().all()
