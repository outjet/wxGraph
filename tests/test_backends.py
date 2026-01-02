from datetime import date
from pathlib import Path

import pytest

from wxgraph.backends.nomads_backend import NomadsBackend
from wxgraph.backends.herbie_backend import HerbieBackend


class DummyResponse:
    status_code = 200

    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        return None


def test_nomads_backend_download(tmp_path, monkeypatch):
    url_log: list[str] = []

    def fake_build(model, run_date, cycle, fh, bbox):
        return ["https://nomads.test/sample.grib"]

    def fake_get(url, timeout):
        url_log.append(url)
        return DummyResponse(b"bytes")

    monkeypatch.setattr("wxgraph.backends.nomads_backend.build_nomads_urls", fake_build)
    monkeypatch.setattr("wxgraph.backends.nomads_backend.requests.get", fake_get)

    backend = NomadsBackend()
    backend.download(
        "gfs",
        [0],
        {"left": -84.0, "right": -79.0, "top": 43.0, "bottom": 40.0},
        tmp_path,
        no_cache=True,
        metadata={"run_date": date(2024, 1, 1), "cycle": 0},
    )

    assert (tmp_path / "gfs_0.grib2").exists()
    assert url_log == ["https://nomads.test/sample.grib"]


def test_herbie_backend_download(tmp_path, monkeypatch):
    class DummyHerbie:
        def __init__(self, date, **kwargs):
            self.save_dir = Path(kwargs.get("save_dir"))

        def download(self, **kwargs):
            self.save_dir.mkdir(parents=True, exist_ok=True)
            target = self.save_dir / "herbiefile.grib2"
            target.write_bytes(b"dummy")
            return target

    monkeypatch.setattr("wxgraph.backends.herbie_backend.Herbie", DummyHerbie)

    backend = HerbieBackend()
    backend.download(
        "gfs",
        [0],
        {"left": -84.0, "right": -79.0, "top": 43.0, "bottom": 40.0},
        tmp_path,
        no_cache=True,
        metadata={"run_date": date(2024, 1, 1), "cycle": 0},
    )

    assert (tmp_path / "gfs_0.grib2").exists()
