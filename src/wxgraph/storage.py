"""Optional Cloud Storage helpers for wxGraph."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterable

LOGGER = logging.getLogger("wxgraph.storage")
LATEST_FILES = ("meteogram_latest.json", "meteogram_latest.png", "meteogram_status.json")
CACHE_DIRNAME = "model_cache"
CACHE_PATTERNS = ("*_latest.json", "*_complete.json")


def _gcs_settings() -> tuple[str, str] | None:
    bucket = os.environ.get("WXGRAPH_GCS_BUCKET")
    if not bucket:
        return None
    prefix = os.environ.get("WXGRAPH_GCS_PREFIX", "latest").strip("/")
    return bucket, prefix


def _blob_name(prefix: str, filename: str) -> str:
    if not prefix:
        return filename
    return f"{prefix}/{filename}"


def _iter_existing_files(output_dir: Path) -> Iterable[Path]:
    for name in LATEST_FILES:
        path = output_dir / name
        if path.exists():
            yield path
    cache_dir = output_dir / CACHE_DIRNAME
    if cache_dir.exists():
        for pattern in CACHE_PATTERNS:
            yield from cache_dir.glob(pattern)


def download_latest_from_gcs(output_dir: Path) -> bool:
    settings = _gcs_settings()
    if not settings:
        return False
    bucket_name, prefix = settings
    try:
        from google.cloud import storage
    except Exception as exc:
        LOGGER.warning("GCS download skipped: %s", exc)
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    downloaded = False
    for filename in LATEST_FILES:
        blob = bucket.blob(_blob_name(prefix, filename))
        if not blob.exists():
            continue
        blob.download_to_filename(output_dir / filename)
        downloaded = True
    cache_dir = output_dir / CACHE_DIRNAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    for pattern in CACHE_PATTERNS:
        for filename in _list_cache_blobs(client, bucket_name, prefix, pattern):
            blob = bucket.blob(filename)
            blob.download_to_filename(cache_dir / Path(filename).name)
            downloaded = True
    if downloaded:
        LOGGER.info("Downloaded latest meteogram files from GCS bucket %s", bucket_name)
    return downloaded


def upload_latest_to_gcs(output_dir: Path) -> None:
    settings = _gcs_settings()
    if not settings:
        return
    bucket_name, prefix = settings
    try:
        from google.cloud import storage
    except Exception as exc:
        LOGGER.warning("GCS upload skipped: %s", exc)
        return

    files = list(_iter_existing_files(output_dir))
    if not files:
        return
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for path in files:
        relative_name = path.relative_to(output_dir).as_posix()
        blob = bucket.blob(_blob_name(prefix, relative_name))
        content_type = "application/json" if path.suffix == ".json" else "image/png"
        blob.upload_from_filename(path, content_type=content_type)
    LOGGER.info("Uploaded latest meteogram files to GCS bucket %s", bucket_name)


def _list_cache_blobs(client, bucket_name: str, prefix: str, pattern: str) -> list[str]:
    prefix_path = _blob_name(prefix, CACHE_DIRNAME)
    blobs = client.list_blobs(bucket_name, prefix=prefix_path)
    results: list[str] = []
    for blob in blobs:
        name = blob.name.split("/")[-1]
        if Path(name).match(pattern):
            results.append(blob.name)
    return results


def get_latest_gcs_metadata() -> dict[str, dict[str, str]]:
    settings = _gcs_settings()
    if not settings:
        return {}
    bucket_name, prefix = settings
    try:
        from google.cloud import storage
    except Exception as exc:
        LOGGER.warning("GCS metadata skipped: %s", exc)
        return {}

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    metadata: dict[str, dict[str, str]] = {}
    for filename in LATEST_FILES:
        blob = bucket.blob(_blob_name(prefix, filename))
        if not blob.exists():
            continue
        blob.reload()
        updated = blob.updated.isoformat() if blob.updated else ""
        metadata[filename] = {
            "bucket": bucket_name,
            "name": blob.name,
            "updated": updated,
        }
    return metadata


def read_gcs_json(name: str) -> dict | None:
    settings = _gcs_settings()
    if not settings:
        return None
    bucket_name, prefix = settings
    try:
        from google.cloud import storage
    except Exception as exc:
        LOGGER.warning("GCS JSON read skipped: %s", exc)
        return None

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(_blob_name(prefix, name))
    if not blob.exists():
        return None
    raw = blob.download_as_bytes()
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        LOGGER.warning("GCS JSON read failed for %s: %s", name, exc)
        return None


def write_gcs_json(name: str, payload: dict) -> None:
    settings = _gcs_settings()
    if not settings:
        return
    bucket_name, prefix = settings
    try:
        from google.cloud import storage
    except Exception as exc:
        LOGGER.warning("GCS JSON write skipped: %s", exc)
        return

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(_blob_name(prefix, name))
    blob.upload_from_string(
        json.dumps(payload, indent=2, sort_keys=True),
        content_type="application/json",
    )
