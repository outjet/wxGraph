FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    WXGRAPH_OUTPUT_DIR=/tmp/wxgraph/output \
    WXGRAPH_WORK_DIR=/tmp/wxgraph/work

RUN apt-get update && apt-get install -y --no-install-recommends \
        libeccodes0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY web /app/web
COPY webapp /app/webapp
COPY gfs_meteogram_kcle.py /app/

RUN pip install --no-cache-dir .
RUN mkdir -p /tmp/wxgraph/output /tmp/wxgraph/work

EXPOSE 8080

CMD ["uvicorn", "webapp.server:app", "--host", "0.0.0.0", "--port", "8080"]
