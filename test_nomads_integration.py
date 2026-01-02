import requests
import pytest
import numpy as np
from pathlib import Path
from datetime import date

from wxgraph.backends.nomads_backend import NomadsBackend
from wxgraph.backends.herbie_backend import HerbieBackend
from wxgraph.backends.base import BackendError

# Define bounding box around KCLE
BBOX = dict(
    bottomlat=41.263,
    toplat=41.697,
    leftlon=-82.099,
    rightlon=-81.521,
)

# A small set of forecast hours
FHOURS = [0, 3, 6]

# GFS subset filter URLs
GFS_FILTER_URLS = [
    (
        "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?"
        "file=gfs.t00z.pgrb2.0p25.f000"
        "&lev_2_m_above_ground=on"
        "&lev_surface=on"
        "&var_TMP=on"
        "&var_DPT=on"
        "&var_SPFH=on"
        "&var_APCP=on"
        "&var_PRMSL=on"
        "&var_GUST=on"
        "&subregion="
        "&leftlon=-82.099"
        "&rightlon=-81.521"
        "&toplat=41.697"
        "&bottomlat=41.263"
        "&dir=/gfs.20251212/00/atmos"
    ),
    (
        "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?"
        "file=gfs.t00z.pgrb2.0p25.f003"
        "&lev_2_m_above_ground=on"
        "&lev_surface=on"
        "&var_TMP=on"
        "&var_DPT=on"
        "&var_SPFH=on"
        "&var_APCP=on"
        "&var_PRMSL=on"
        "&var_GUST=on"
        "&subregion="
        "&leftlon=-82.099"
        "&rightlon=-81.521"
        "&toplat=41.697"
        "&bottomlat=41.263"
        "&dir=/gfs.20251212/00/atmos"
    ),
    (
        "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?"
        "file=gfs.t00z.pgrb2.0p25.f006"
        "&lev_2_m_above_ground=on"
        "&lev_surface=on"
        "&var_TMP=on"
        "&var_DPT=on"
        "&var_SPFH=on"
        "&var_APCP=on"
        "&var_PRMSL=on"
        "&var_GUST=on"
        "&subregion="
        "&leftlon=-82.099"
        "&rightlon=-81.521"
        "&toplat=41.697"
        "&bottomlat=41.263"
        "&dir=/gfs.20251212/00/atmos"
    ),
]


def download_grib(url: str, out_path: Path, *, no_cache: bool = False) -> None:
    """Simple helper to download a GRIB file to disk."""
    import requests
    if out_path.exists() and not no_cache:
        return
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)


@pytest.mark.parametrize("backend_cls", [NomadsBackend, HerbieBackend])
def test_gfs_filter_fetch_and_extract(tmp_path, backend_cls):
    """
    Integration test: download 3 small GFS GRIB subsets and make sure
    our backends can extract a point record for each one.
    """

    backend = backend_cls()

    # If Herbie isn't available, skip HerbieBackend tests
    if backend_cls is HerbieBackend and not backend._is_available():
        pytest.skip("Herbie not installed or not available in this environment")

    work_dir = tmp_path / "gfs"
    work_dir.mkdir()

    # Download small GRIBs for each URL
    for i, url in enumerate(GFS_FILTER_URLS):
        out_path = work_dir / f"gfs_f{i:03d}.grib2"
        try:
            download_grib(url, out_path, no_cache=True)
        except requests.HTTPError as exc:
            pytest.skip(f"Remote subset unavailable: {exc}")

        # Extract a point near KCLE
        rec = backend.extract_point(out_path, lat=41.48, lon=-81.81, fh=FHOURS[i])

        # Assert that some basic fields exist and are numeric
        assert "temp_f" in rec and isinstance(rec["temp_f"], float)
        assert not np.isnan(rec["temp_f"])
        assert "qpf_in" in rec or "qpf_in_raw" in rec
        assert "wind10m_mph" in rec

        # Optionally check dewpoint or RH
        assert "rh_percent" in rec
