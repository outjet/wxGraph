from datetime import date
from pathlib import Path
from typing import Sequence

import pandas as pd

from wxgraph.backends.base import FetchBackend
from wxgraph.models.gfs import GFS


class DummyBackend(FetchBackend):
    def __init__(self):
        self.download_calls = []

    def download(
        self,
        model: str,
        fhours: Sequence[int],
        bbox: dict[str, float],
        outdir: Path,
        *,
        no_cache: bool,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.download_calls.append((model, tuple(fhours), bbox, metadata))

    def extract_point(self, path: Path, lat: float, lon: float, fh: int) -> dict[str, object]:
        return {
            "valid_time": pd.Timestamp("2024-01-01"),
            "temp_c": 0.0,
            "wind10m_ms": 0.0,
            "rh_percent": 0.0,
            "qpf_in_raw": 0.0,
        }


def test_model_enqueue_fetch(tmp_path):
    backend = DummyBackend()
    model = GFS(
        run_date=date(2024, 1, 1),
        cycle=0,
        lat=41.48,
        lon=-81.81,
        fhours=[0],
        work_dir=tmp_path,
    )
    model.fetch(backend, no_cache=True)
    df = model.to_dataframe(backend)
    assert len(backend.download_calls) == 1
    assert not df.empty
