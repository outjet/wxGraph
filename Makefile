PYTHON ?= python3
APP ?= gfs_meteogram_kcle.py
OUTDIR ?= $(shell grep -E '^WXGRAPH_OUTPUT_DIR' .env 2>/dev/null | cut -d= -f2 | tr -d '\r' || echo gfs_meteogram_output)
MODELS ?= $(shell grep -E '^WXGRAPH_MODELS' .env 2>/dev/null | cut -d= -f2 | tr -d '\r' || echo gfs,nam,rap,hrrr,gefs,aigfs,hgefs)
HOURS ?= $(shell grep -E '^WXGRAPH_HOURS' .env 2>/dev/null | cut -d= -f2 | tr -d '\r' || echo 0-168:3)
SITE ?= $(shell grep -E '^WXGRAPH_SITE' .env 2>/dev/null | cut -d= -f2 | tr -d '\r' || echo KCLE)

.PHONY: install fetch dev serve lint test clean

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .[dev]

fetch:
	$(PYTHON) $(APP) --models $(MODELS) --hours $(HOURS) --location-label $(SITE) --merge-csv --write-json

serve:
	uvicorn webapp.server:app --reload --host 0.0.0.0 --port 8000

lint:
	ruff check .

lint-fix:
	ruff check . --fix

test:
	pytest

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
