#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "$ROOT_DIR"
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi
python gfs_meteogram_kcle.py --models "${WXGRAPH_MODELS:-gfs,nam,rap,hrrr}" --hours "${WXGRAPH_HOURS:-0-168:3}" --location-label "${WXGRAPH_SITE:-KCLE}" --merge-csv --write-json
uvicorn webapp.server:app --reload --host 0.0.0.0 --port "${WXGRAPH_PORT:-8000}"
