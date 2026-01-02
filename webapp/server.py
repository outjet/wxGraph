import json
import os
import sys
import threading
from datetime import datetime, date
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = BASE_DIR / "web"
for candidate in (BASE_DIR / "src", BASE_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from gfs_meteogram_kcle import choose_default_cycle, find_latest_available_gfs_run, get_nomads_utc_now
from wxgraph.backends.nomads_backend import NomadsBackend
from wxgraph.config import (
    DEFAULT_FHOURS,
    DEFAULT_LAT,
    DEFAULT_LOCATION_LABEL,
    DEFAULT_LON,
    get_output_dir,
    get_work_dir,
)
from wxgraph.runner import MeteogramRunner
from wxgraph.storage import download_latest_from_gcs, get_latest_gcs_metadata, upload_latest_to_gcs

LATEST_NAME = "meteogram_latest.json"
RUN_LOCK = threading.Lock()

app = FastAPI()
app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")


def _latest_json_path() -> Path:
    candidates = sorted(
        get_output_dir().glob(f"**/{LATEST_NAME}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        if download_latest_from_gcs(get_output_dir()):
            candidates = sorted(
                get_output_dir().glob(f"**/{LATEST_NAME}"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
    if not candidates:
        raise HTTPException(status_code=404, detail=f"{LATEST_NAME} not found")
    return candidates[0]


@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/meteogram_latest.json")
async def legacy_json():
    """Legacy path retained for backwards compatibility."""

    return FileResponse(_latest_json_path())


@app.get("/api/meteogram/latest")
async def latest_json():
    path = _latest_json_path()
    return JSONResponse(content=json.loads(path.read_text()))


@app.get("/api/health")
async def health():
    try:
        latest = _latest_json_path()
        status = {"status": "ok", "latest_json": str(latest)}
    except HTTPException as exc:
        status = {"status": "degraded", "detail": exc.detail}
    return status


@app.get("/api/meteogram/status")
async def meteogram_status():
    status_path = get_output_dir() / "meteogram_status.json"
    status: dict[str, object] = {
        "output_dir": str(get_output_dir()),
        "latest_json": None,
        "latest_json_mtime": None,
        "gcs": get_latest_gcs_metadata(),
        "model_status": None,
    }
    if status_path.exists():
        try:
            status["model_status"] = json.loads(status_path.read_text())
        except json.JSONDecodeError:
            status["model_status"] = {"error": "Invalid status JSON"}
    try:
        latest = _latest_json_path()
    except HTTPException:
        return status
    status["latest_json"] = str(latest)
    status["latest_json_mtime"] = datetime.utcfromtimestamp(latest.stat().st_mtime).isoformat()
    return status


@app.get("/app.js")
async def app_js():
    return FileResponse(WEB_DIR / "app.js")


@app.get("/style.css")
async def style_css():
    return FileResponse(WEB_DIR / "style.css")


def _determine_latest_run() -> tuple[date, int]:
    now = datetime.utcnow()
    nomads_now = get_nomads_utc_now()
    if nomads_now is not None:
        now = nomads_now
    run_cycle = choose_default_cycle(now)
    candidate = find_latest_available_gfs_run(now.date(), run_cycle)
    if candidate is None:
        raise RuntimeError("Could not find a recent GFS cycle to run.")
    return candidate


def _run_meteogram(run_date: date, cycle: int) -> None:
    with RUN_LOCK:
        runner = MeteogramRunner(
            run_date=run_date,
            cycle=cycle,
            lat=DEFAULT_LAT,
            lon=DEFAULT_LON,
            fhours=DEFAULT_FHOURS,
            work_dir=get_work_dir(),
            output_dir=get_output_dir(),
            location_label=DEFAULT_LOCATION_LABEL,
            backend=NomadsBackend(),
        )
        runner.run()
        upload_latest_to_gcs(get_output_dir())


def _auto_refresh_loop(interval_minutes: int) -> None:
    """Run the meteogram pipeline on a fixed interval in the background."""

    while True:
        try:
            run_date, cycle = _determine_latest_run()
            _run_meteogram(run_date, cycle)
        except Exception as exc:
            print(f"[wxgraph] auto-refresh failed: {exc}")
        time.sleep(interval_minutes * 60)


def _require_run_token(request: Request) -> None:
    token = os.environ.get("WXGRAPH_RUN_TOKEN")
    if not token:
        return
    supplied = request.headers.get("x-run-token")
    if supplied != token:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.post("/api/meteogram/run", status_code=202)
async def trigger_run(request: Request, background_tasks: BackgroundTasks):
    _require_run_token(request)
    try:
        run_date, cycle = _determine_latest_run()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    background_tasks.add_task(_run_meteogram, run_date, cycle)
    return {
        "status": "accepted",
        "detail": f"Queued run {run_date:%Y-%m-%d} t{cycle:02d}z",
        "run_date": run_date.strftime("%Y%m%d"),
        "cycle": cycle,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
