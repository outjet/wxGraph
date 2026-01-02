# wxGraph

wxGraph generates a multi-model meteogram for a single site (KCLE by default), saves the derived
weather tables/plots, and exposes a lightweight Plotly web UI for interactive review. The tool pulls
point data from NOMADS, merges common derived fields (temperature, precip, snowfall, icing risk), and
keeps the latest JSON/PNG bundle ready for a tiny FastAPI server to serve.

## Features
- Downloads multiple point forecasts (GFS/NAM/HRRR/RAP/GEFS/AIGFS/HGEFS) and backfills to the most
  recently available cycle automatically.
- Computes road-icing diagnostics, merged CSV/JSON, and a static PNG meteogram.
- Ships a zero-build frontend (`web/`) that consumes the merged JSON and renders Plotly panels with
  synced interactions.
- FastAPI server (`webapp/server.py`) serves the static assets plus `/api/meteogram/latest` and
  `/api/health` endpoints for production use.
- Includes helper scripts, Make targets, and CI/test scaffolding for production hardening.

## Quickstart
1. **Clone & install**
   ```bash
   git clone https://github.com/outjet/wxGraph.git
   cd wxGraph
   cp .env.example .env   # edit lat/lon, models, etc.
   make install            # pip install -e .[dev]
   ```
2. **Fetch the latest meteogram bundle**
   ```bash
   PYTHONPATH=src ./venv/bin/python -m wxgraph.cli --models gfs,nam,rap --run-date 20240101 --cycle 0 --fhours 0-24:3
   ```
   The runner stores `output/meteogram_latest.json` and `output/meteogram_latest.png` ready for the web UI; `make fetch` still exists for compatibility but the CLI now drives the modular stack.
3. **Serve the frontend/API locally**
   ```bash
   make serve
   # open http://localhost:8000
   ```
   Use `scripts/dev.sh` to fetch + run a reloadable server in one go.

## Configuration
wxGraph reads defaults from CLI flags _or_ environment variables defined in `.env`:

| Variable | Description |
| --- | --- |
| `WXGRAPH_LAT`, `WXGRAPH_LON` | Point latitude/longitude |
| `WXGRAPH_SITE` | Label used in filenames and UI |
| `WXGRAPH_OUTPUT_DIR` | Root directory for model caches, PNG, CSV, JSON |
| `WXGRAPH_MODELS` | Comma-separated model list (`gfs,nam,rap,hrrr,gefs,aigfs,hgefs`) |
| `WXGRAPH_HOURS` | Forecast hour spec (`start-end:step`) |
| `WXGRAPH_LOG_LEVEL` | Python log level (`INFO`, `DEBUG`, …) |
| `WXGRAPH_REFRESH_MINUTES` | Hint for external schedulers |
| `WXGRAPH_GCS_BUCKET` | Optional GCS bucket for persisting latest JSON/PNG |
| `WXGRAPH_GCS_PREFIX` | Optional prefix for GCS objects (default: `latest`) |
| `WXGRAPH_LCR_ALERT_ENABLED` | Set `1` to enable LCR Pushover alerts |
| `WXGRAPH_LCR_ALERT_MODEL` | Model to scan for LCR alerts (default: `HRRR`) |
| `WXGRAPH_LCR_ALERT_HOURS_AHEAD` | Hours of forecast to scan (default: `12`) |
| `WXGRAPH_LCR_ALERT_THRESHOLD` | Minimum LCR score to alert (default: `5`) |
| `WXGRAPH_LCR_ALERT_COOLDOWN_HOURS` | Cooldown window between alerts (default: `4`) |
| `WXGRAPH_LCR_ALERT_STATE_NAME` | GCS object name for alert state (default: `lcr_alert_state.json`) |
| `WXGRAPH_PUSHOVER_APP_TOKEN` | Pushover application token |
| `WXGRAPH_PUSHOVER_USER_KEY` | Pushover user key |
| `WXGRAPH_RUN_TOKEN` | Optional token required for `POST /api/meteogram/run` |
| `WXGRAPH_SNOW_RATIO` | Snow ratio method (`10to1` or `baxter`) |
| `WXGRAPH_BLEND_PERIODS` | Forecast steps to blend when stitching runs |

You can still override parameters with CLI flags. The new runner exposes the same options:
```bash
PYTHONPATH=src ./venv/bin/python -m wxgraph.cli --lat 41.3 --lon -81.9 --models gfs,rap --fhours 0-36:3
```

## API-triggered runner
POST `/api/meteogram/run` enqueues the latest cycle via FastAPI without blocking the request. It uses the same lat/lon/bbox defaults as the CLI, downloads every integrated model, runs the pipelines, and writes `meteogram_latest.{json,png}` under `WXGRAPH_OUTPUT_DIR` (defaults to `gfs_meteogram_output/`). The web UI keeps polling `/api/meteogram/latest`.

## Status + caching behavior
- `/api/meteogram/status` reports the latest local JSON timestamp and (when configured) GCS object metadata.
- Each model caches its latest output; if a new run is partial, the system backfills missing valid times from the last complete run for that model.
- `/api/meteogram/status` includes per-model success/missing-hour details via `meteogram_status.json`.

## Data Flow
1. `gfs_meteogram_kcle.py` downloads GRIB files per model/hour. If a requested run is not yet on
   NOMADS, it automatically checks previous cycles.
2. Each model DataFrame is validated, augmented with icing/wetbulb diagnostics, and saved to CSV when
   `--write-csv` is enabled.
3. `--merge-csv/--write-json` create `meteogram_latest.json` with a wide schema (per-model columns)
   for the web UI.
4. `webapp/server.py` serves `/` (static frontend) and `/api/meteogram/latest` (latest JSON). The
   server scans `WXGRAPH_OUTPUT_DIR` for the newest JSON if multiple runs exist.

## Icing / LCR Diagnostics
The backend now computes a heuristic loss-of-control risk (LCR) index and related icing fields for
every model/time. Calculations happen in Python (see `wxgraph/icing.py`) so the frontend simply
remaps the JSON fields.

### Fields emitted per model (examples use `GFS_` prefix)
- `GFS_temp_c`, `GFS_temp_f`, `GFS_temp_hist6h_c`, `GFS_temp_hist6h_f`
- `GFS_rh_pct`, `GFS_dp_c`, `GFS_dp_f`, `GFS_wb_c`, `GFS_wb_f`
- `GFS_qpf_in_accum`, `GFS_ipf_in` (per step in inches), `GFS_ipf_in_per_hr` (per-hour normalized)
- `GFS_snow_in_accum`, `GFS_snowsfc_in`, `GFS_snowsfc_in_per_hr`
- `GFS_cip_inph`, `GFS_bfp_inph`, `GFS_nfp_inph`, `GFS_afp_inph` (critical/near-freezing buckets)
  and the per-step equivalents (`*_cip_in`, etc.)
- `GFS_lcr`, `GFS_lcron` (baseline-flag), `GFS_dt_hours`, `GFS_gust_mph`, `GFS_cloud_pct`

### Units and usage
- Accumulated precipitation/snow are inches; per-step values represent the model timestep.
- Per-hour values (`*_in_per_hr`, `*_inph`) divide by `*_dt_hours` so they can be compared against
  the thresholds in the original NOMADS scripts.
- `*_dt_hours` documents the timestep so external tools can convert between per-step/per-hour.

### Severity mapping (used by upcoming frontend strip)
- `0`: None
- `1–2`: Low
- `3–4`: Elevated
- `5–6`: High
- `>=7`: Extreme

### Caveats
- We approximate dewpoint via Bolton (1980) when explicit fields are missing; wet bulb uses the
  Stull formulation and assumes RH exists.
- Snow accumulations use the existing `snow_acc_in` (compacted) when available, otherwise a simple
  cumulative sum of per-step snowfall; treat units accordingly.
- Without explicit freezing-rain categorical flags, LCR increments for freezing rain rely on the
  supplied columns (currently none), so freezing-drizzle detection remains conservative.
- Cloud cover defaults to 100% when the model does not provide `tcdc`.
- LCR is intended for situational awareness only; it does not account for local treatment,
  bridges/overpasses, or sub-grid microclimates.

## Commands & Scripts
- `make install` – install dependencies in editable mode.
- `PYTHONPATH=src ./venv/bin/python -m wxgraph.cli --models gfs,nam --run-date 20240101 --cycle 0 --fhours 0-24:3` – run the modular fetch/normalize/publish workflow.
- `make fetch` – legacy target that still exists but now wraps the same pipeline.
- `curl -X POST http://localhost:8000/api/meteogram/run` – trigger the API-backed runner (outputs written to `WXGRAPH_OUTPUT_DIR`).
- `make serve` – start the FastAPI server (reload disabled).
- `make dev` – see `scripts/dev.sh` (fetch + `uvicorn --reload`).
- `make lint` / `make lint-fix` – Ruff static analysis.
- `make test` – pytest suite.

Helper scripts mirror the Make targets and respect `.env` defaults:
- `scripts/fetch.sh`
- `scripts/serve.sh`
- `scripts/dev.sh`

## Deployment
### systemd timers (recommended example)
Systemd units live under `systemd/`:
- `wxgraph-fetch.service` / `wxgraph-fetch.timer` – run the fetcher hourly (honors `.env`).
- `wxgraph-serve.service` – run `uvicorn webapp.server:app` as a long-lived service.

Install example:
```bash
sudo cp systemd/wxgraph-fetch.* /etc/systemd/system/
sudo cp systemd/wxgraph-serve.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now wxgraph-fetch.timer wxgraph-serve.service
```
Tune the timer/ExecStart to match your site’s desired cadence.

### Docker (optional)
See `Dockerfile` for a production image that runs the FastAPI service on port 8080.
`docs/gcp.md` walks through a Cloud Run deployment.

## Tests & CI
- Unit tests live under `tests/`; run `pytest` locally or via `make test`.
- Integration tests download real NOMADS subsets; set `WXGRAPH_RUN_INTEGRATION=1` to enable the `tests/test_nomads_integration.py` and `tests/test_hrrr_integration.py` suites.
- `.github/workflows/ci.yml` executes Ruff + pytest on every push/PR.

## Troubleshooting
| Symptom | Likely Cause & Fix |
| --- | --- |
| `404/403` for every model | Requested cycle not yet on NOMADS; the tool automatically steps back, but check log timestamps / run `--date YYYYMMDD --cycle HH` explicitly. |
| Plotly shows only one panel | Ensure `meteogram_latest.json` exists and web server has read access. Use browser dev tools to inspect `/api/meteogram/latest`. |
| FastAPI server returns 404 for JSON | Confirm `WXGRAPH_OUTPUT_DIR` contains `meteogram_latest.json` and the service user has permission to read it. |
| NOMADS sporadically fails | Rerun `PYTHONPATH=src ./venv/bin/python -m wxgraph.cli --models gfs,nam --run-date YYYYMMDD --cycle 0` to refresh the bundle; consider adding retry/backoff or caching the last good JSON (on the roadmap). |

## License & Contributing
- Licensed under MIT (see `LICENSE`).
- Contributions welcome! File issues/PRs with clear steps to reproduce and ensure `make lint test` is
  clean.
