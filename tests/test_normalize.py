import pandas as pd
import pytest

from wxgraph.pipeline.normalize import normalize


def test_normalize_converts_units_and_times():
    df = pd.DataFrame(
        {
            "valid_time": ["2024-01-01T00:00", "2024-01-01T03:00"],
            "temp_k": [273.15, 280.15],
            "wind10m_ms": [5.0, 10.0],
            "rh": [80.0, 60.0],
        }
    )
    normalized = normalize(df)
    assert pd.api.types.is_datetime64_any_dtype(normalized["valid_time"])
    assert normalized["temp_c"].tolist() == pytest.approx([0.0, 7.0])
    assert normalized["temp_f"].tolist() == pytest.approx([32.0, 44.6])
    assert normalized["wind10m_mph"].tolist() == pytest.approx([11.1847, 22.3694], rel=1e-6)
    assert normalized["rh_percent"].tolist() == pytest.approx([80.0, 60.0])
    assert normalized["model"].iloc[0] == "unknown"
