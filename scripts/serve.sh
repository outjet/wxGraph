#!/usr/bin/env bash
set -euo pipefail
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi
uvicorn webapp.server:app --host 0.0.0.0 --port "${WXGRAPH_PORT:-8000}" "$@"
