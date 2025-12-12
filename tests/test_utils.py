import numpy as np

from gfs_meteogram_kcle import limit_forecast_hours, precip_temp_partitions, wet_bulb_f


def test_limit_forecast_hours_truncates_and_warns(capfd):
    hours = [0, 3, 6, 48, 72]
    result = limit_forecast_hours("hrrr", hours)
    assert max(result) == 48
    assert 72 not in result
    # ensures message emitted once
    out = capfd.readouterr().out
    assert "HRRR" in out
    assert "truncating" in out


def test_precip_temp_partitions_threshold():
    temps = np.array([28.0, 30.0, 34.0, 40.0])
    qpf = np.array([0.1, 0.02, 0.05, 0.005])
    parts = precip_temp_partitions(temps, qpf)
    assert np.isclose(parts["cip_in"][0], 0.1)
    # 0.02 should remain BFP but the final value should be 0.02 (>= 0.003)
    assert np.isclose(parts["bfp_in"][1], 0.02)
    # 0.005 > threshold but falls into AFP bucket
    assert np.isclose(parts["afp_in"][3], 0.005)


def test_wet_bulb_handles_missing_values():
    temps = np.array([32.0, np.nan, 50.0])
    rhs = np.array([90.0, 40.0, np.nan])
    result = wet_bulb_f(temps, rhs)
    assert not np.isnan(result[0])
    assert np.isnan(result[1])
    assert np.isnan(result[2])
