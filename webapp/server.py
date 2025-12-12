import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = BASE_DIR / "web"
default_output = BASE_DIR / "gfs_meteogram_output"
raw_out = os.environ.get("WXGRAPH_OUTPUT_DIR")
OUT_ROOT = Path(raw_out).expanduser() if raw_out else default_output
if not OUT_ROOT.is_absolute():
    OUT_ROOT = (BASE_DIR / OUT_ROOT).resolve()
LATEST_NAME = "meteogram_latest.json"

app = FastAPI()
app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")


def _latest_json_path() -> Path:
    candidates = sorted(
        OUT_ROOT.glob(f"**/{LATEST_NAME}"),
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


@app.get("/app.js")
async def app_js():
    return FileResponse(WEB_DIR / "app.js")


@app.get("/style.css")
async def style_css():
    return FileResponse(WEB_DIR / "style.css")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
