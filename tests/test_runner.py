from datetime import date
from pathlib import Path

import pandas as pd

from wxgraph.backends.base import FetchBackend
from wxgraph.runner import MeteogramRunner


class DummyBackend(FetchBackend):
    def __init__(self):
        self.download_calls: list[tuple[str, tuple[int, ...]]] = []

    def download(
        self,
        model: str,
        fhours: list[int],
        bbox,
        outdir: Path,
        *,
        no_cache: bool,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.download_calls.append((model, tuple(fhours)))

    def extract_point(self, path: Path, lat: float, lon: float, fh: int) -> dict[str, object]:
        return {
            "valid_time": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=fh),
            "temp_c": 2.0 + fh,
            "wind10m_ms": 5.0,
            "rh_percent": 80.0,
            "qpf_in_raw": 0.1 * (fh + 1),
            "gust_mph": 7.0,
        }


def test_runner_exports_json_and_png(tmp_path):
    backend = DummyBackend()
    plot_calls: list[Path] = []

    def fake_plot(model_dfs, location_label, run_label, out_path: Path) -> None:
        plot_calls.append(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("fake plot")

    runner = MeteogramRunner(
        models=["gfs"],
        lat=41.48,
        lon=-81.81,
        run_date=date(2024, 1, 1),
        cycle=0,
        fhours=[0, 3],
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "out",
        backend=backend,
        plotter=fake_plot,
        no_cache=True,
    )

    result = runner.run()
    assert result["json"].exists()
    assert plot_calls
    assert result["png"].exists()
    data = pd.read_json(result["json"])
    assert not data.empty
