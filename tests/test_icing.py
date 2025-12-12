import numpy as np
import pandas as pd

from wxgraph.icing import add_icing_fields


def _build_df(freq_hours: int, temps_c: list[float], qpf_mm_accum: list[float], snow_mm_accum: list[float], rh: float = 95.0):
    times = pd.date_range("2024-01-01", periods=len(temps_c), freq=f"{freq_hours}H")
    qpf_acc = np.array(qpf_mm_accum)
    snow_acc = np.array(snow_mm_accum)
    return pd.DataFrame(
        {
            "valid_time": times,
            "temp_k": np.array(temps_c) + 273.15,
            "rh": rh,
            "dp_k": np.array(temps_c) + 271.15,
            "qpf_mm": qpf_acc,
            "snow_mm": snow_acc,
            "gust_ms": 5.0,
            "cloud_pct": 50.0,
        }
    )


def test_kelvin_and_wetbulb_conversions():
    df = _build_df(1, [0.0, -1.0], [0.0, 2.54], [0.0, 0.0])
    result = add_icing_fields(
        df,
        prefix="test",
        time_col="valid_time",
        temp_k_col="temp_k",
        rh_pct_col="rh",
        dp_k_col="dp_k",
        qpf_accum_mm_col="qpf_mm",
        snow_accum_mm_col="snow_mm",
        gust_ms_col="gust_ms",
        cloud_pct_col="cloud_pct",
        latitude=41.0,
    )
    temp_c = result["TEST_temp_c"].to_numpy()
    assert np.allclose(temp_c, [0.0, -1.0], atol=1e-3)
    temp_f = result["TEST_temp_f"].to_numpy()
    assert np.allclose(temp_f, [32.0, 30.2], atol=0.1)
    wb_f = result["TEST_wb_f"].to_numpy()
    assert np.isfinite(wb_f).all()
    assert result["TEST_cloud_pct"].iloc[0] == 50.0


def test_decumulation_and_dt_hours():
    df = _build_df(3, [1.0, 1.5, 2.0], [0.0, 3.0, 6.0], [0.0, 0.0, 0.0])
    result = add_icing_fields(
        df,
        prefix="test",
        time_col="valid_time",
        temp_k_col="temp_k",
        rh_pct_col="rh",
        dp_k_col="dp_k",
        qpf_accum_mm_col="qpf_mm",
        snow_accum_mm_col="snow_mm",
        gust_ms_col="gust_ms",
        cloud_pct_col="cloud_pct",
    )
    assert result["TEST_dt_hours"].iloc[0] == 3.0
    ipf = result["TEST_ipf_in"].to_numpy()
    assert np.allclose(ipf, [0.0, 0.1181103, 0.1181103], atol=1e-6)
    ipf_hr = result["TEST_ipf_in_per_hr"].to_numpy()
    assert np.allclose(ipf_hr[1:], ipf[1:] / 3.0, atol=1e-6)


def test_lcr_baseline_and_increments():
    temps_c = [-5.0, -4.0]
    qpf_steps = [0.0, 15.0]
    snow_steps = [0.0, 0.0]
    df = _build_df(1, temps_c, qpf_steps, snow_steps, rh=90.0)
    result = add_icing_fields(
        df,
        prefix="test",
        time_col="valid_time",
        temp_k_col="temp_k",
        rh_pct_col="rh",
        dp_k_col="dp_k",
        qpf_accum_mm_col="qpf_mm",
        snow_accum_mm_col="snow_mm",
        gust_ms_col="gust_ms",
        cloud_pct_col="cloud_pct",
        latitude=30.0,
    )
    lcr = result["TEST_lcr"].to_numpy()
    assert lcr[1] >= 4  # baseline plus increments
    assert result["TEST_lcron"].iloc[1] == 1


def test_cip_buckets_and_freezing_fog():
    temps_c = [-1.0, -2.0]
    qpf_steps = [0.0, 0.0]
    snow_steps = [0.0, 0.0]
    df = _build_df(1, temps_c, qpf_steps, snow_steps, rh=99.0)
    result = add_icing_fields(
        df,
        prefix="test",
        time_col="valid_time",
        temp_k_col="temp_k",
        rh_pct_col="rh",
        dp_k_col="dp_k",
        qpf_accum_mm_col="qpf_mm",
        snow_accum_mm_col="snow_mm",
        gust_ms_col="gust_ms",
        cloud_pct_col=None,
    )
    assert result["TEST_cloud_pct"].iloc[0] == 100.0  # default fallback
    assert result["TEST_cip_inph"].max() == 0.0
    lcr = result["TEST_lcr"].to_numpy()
    assert lcr[0] >= 1  # freezing fog factor
    assert np.all(result["TEST_lcr"] <= 3)  # clear-sky cap kicks in
