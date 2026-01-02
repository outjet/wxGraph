import pandas as pd

from wxgraph.pipeline.merge import merge_models


def test_merge_models_combines_and_labels_sources():
    left = pd.DataFrame({"valid_time": pd.date_range("2024-01-01", periods=2), "temp_c": [0.0, 1.0]})
    right = pd.DataFrame({"valid_time": pd.date_range("2024-01-02", periods=2), "temp_c": [2.0, 3.0]})
    merged = merge_models({"gfs": left, "hrrr": right})
    assert "source_model" in merged.columns
    assert set(merged["source_model"].unique()) == {"gfs", "hrrr"}
    assert merged["valid_time"].is_monotonic_increasing
