import requests
import pytest
import numpy as np
from pathlib import Path
from wxgraph.backends.nomads_backend import NomadsBackend
from wxgraph.backends.herbie_backend import HerbieBackend

# Bounding box around KCLE (~15 miles / ~25 km)
BBOX = dict(
    bottomlat=41.263,
    toplat=41.697,
    leftlon=-82.099,
    rightlon=-81.521,
)

# Small forecast hours (few points just for integration)
FH_HRRR = [3, 6]  # HRRR files use 2-digit fh (03, 06 …)
FH_RAP  = [3, 6]

# HRRR filter URLs for surface variables near KCLE
HRRR_FILTER_URLS = [
    (
        "https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl"
        "?file=hrrr.t00z.wrfsfcf03.grib2"
        "&subregion="
        "&leftlon=-82.099"
        "&rightlon=-81.521"
        "&toplat=41.697"
        "&bottomlat=41.263"
        "&var_TMP=on"
        "&var_DPT=on"
        "&var_SPFH=on"
        "&var_APCP=on"
        "&var_PRMSL=on"
        "&var_GUST=on"
    ),
    (
        "https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl"
        "?file=hrrr.t00z.wrfsfcf06.grib2"
        "&subregion="
        "&leftlon=-82.099"
        "&rightlon=-81.521"
        "&toplat=41.697"
        "&bottomlat=41.263"
        "&var_TMP=on"
        "&var_DPT=on"
        "&var_SPFH=on"
        "&var_APCP=on"
        "&var_PRMSL=on"
        "&var_GUST=on"
    ),
]

# RAP filter URLs for surface variables near KCLE
RAP_FILTER_URLS = [
    (
        "https://nomads.ncep.noaa.gov/cgi-bin/filter_rap_2d.pl"
        "?file=rap.t00z.awp130pgrbf03.grib2"
        "&subregion="
        "&leftlon=-82.099"
        "&rightlon=-81.521"
        "&toplat=41.697"
        "&bottomlat=41.263"
        "&var_TMP=on"
        "&var_DPT=on"
        "&var_SPFH=on"
        "&var_APCP=on"
        "&var_PRMSL=on"
        "&var_GUST=on"
    ),
    (
        "https://nomads.ncep.noaa.gov/cgi-bin/filter_rap_2d.pl"
        "?file=rap.t00z.awp130pgrbf06.grib2"
        "&subregion="
        "&leftlon=-82.099"
        "&rightlon=-81.521"
        "&toplat=41.697"
        "&bottomlat=41.263"
        "&var_TMP=on"
        "&var_DPT=on"
        "&var_SPFH=on"
        "&var_APCP=on"
        "&var_PRMSL=on"
        "&var_GUST=on"
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


@pytest.mark.parametrize(
    "backend_cls, urls, fh_list",
    [
        (NomadsBackend, HRRR_FILTER_URLS, FH_HRRR),
        (HerbieBackend, HRRR_FILTER_URLS, FH_HRRR),
        (NomadsBackend, RAP_FILTER_URLS, FH_RAP),
        (HerbieBackend, RAP_FILTER_URLS, FH_RAP),
    ],
)
def test_hrrr_rap_fetch_extract(tmp_path, backend_cls, urls, fh_list):
    """
    Integration test: download small HRRR & RAP GRIB subsets
    and make sure backends can extract a point record for each one.
    """

    backend = backend_cls()

    # Skip Herbie tests if Herbie is unavailable
    if backend_cls is HerbieBackend and not backend._is_available():
        pytest.skip("Herbie not installed or not available in this environment")

    work_dir = tmp_path / backend_cls.__name__
    work_dir.mkdir()

    for i, url in enumerate(urls):
        out_path = work_dir / f"file_{i:03d}.grib2"
        try:
            download_grib(url, out_path, no_cache=True)
        except requests.HTTPError as exc:
            pytest.skip(f"Remote subset unavailable: {exc}")

        # Extract the point near KCLE
        fh = fh_list[i]
        rec = backend.extract_point(out_path, lat=41.48, lon=-81.81, fh=fh)

        # Assert the basic fields exist and look reasonable
        assert "temp_f" in rec and isinstance(rec["temp_f"], float)
        assert not np.isnan(rec["temp_f"])
        assert "qpf_in" in rec or "qpf_in_raw" in rec
        assert "wind10m_mph" in rec
        assert "rh_percent" in rec
