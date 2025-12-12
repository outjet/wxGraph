#!/usr/bin/env bash
set -euo pipefail
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi
python gfs_meteogram_kcle.py \
  --models "${WXGRAPH_MODELS:-gfs,nam,rap,hrrr}" \
  --hours "${WXGRAPH_HOURS:-0-168:3}" \
  --location-label "${WXGRAPH_SITE:-KCLE}" \
  --merge-csv \
  --write-json "$@"
