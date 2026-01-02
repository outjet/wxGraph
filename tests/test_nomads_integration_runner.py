import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wxgraph.backends.nomads_backend import NomadsBackend
from wxgraph.pipeline.normalize import normalize

LAT = 41.48
LON = -81.81
RUN_DATE = date(2025, 12, 12)
CYCLE = 0
BBOX = {"left": -82.099, "right": -81.521, "top": 41.697, "bottom": 41.263}
FHOURS = [0, 3, 6, 9, 12, 15]
BBOX = {"left": -82.099, "right": -81.521, "top": 41.697, "bottom": 41.263}


def _integration_enabled() -> bool:
    return os.environ.get("WXGRAPH_RUN_INTEGRATION", "").lower() in {"1", "true", "yes"}


def _require_integration():
    if not _integration_enabled():
        pytest.skip("Integration tests disabled; set WXGRAPH_RUN_INTEGRATION=1 to enable")


def _validate_record(record: dict[str, float]):
    assert not np.isnan(record.get("temp_c", np.nan))
    assert (
        not np.isnan(record.get("rh_percent", np.nan))
        or not np.isnan(record.get("tdew_c", np.nan))
    )
    assert not np.isnan(record.get("qpf_in_raw", np.nan))
    wind = record.get("wind10m_ms", np.nan)
    gust = record.get("gust_mph", np.nan)
    assert not np.isnan(wind) or not np.isnan(gust)


@pytest.mark.integration
@pytest.mark.parametrize("model_key", ["gfs", "hrrr", "rap"])
def test_nomads_subset_returns_required_fields(model_key: str, tmp_path: Path):
    _require_integration()

    backend = NomadsBackend()
    out_dir = tmp_path / model_key
    backend.download(
        model_key,
        FHOURS,
        BBOX,
        out_dir,
        no_cache=True,
        metadata={"run_date": RUN_DATE, "cycle": CYCLE},
    )

    records: list[dict[str, float]] = []
    for fh in FHOURS:
        path = out_dir / f"gfs_{fh}.grib2"
        assert path.exists(), f"Missing GRIB for f{fh}"
        record = backend.extract_point(path, LAT, LON, fh)
        _validate_record(record)
        records.append(record)

    df = pd.DataFrame.from_records(records)
    normalized = normalize(df)
    assert not normalized.empty
