# Deploying wxGraph to Google Cloud Run

This guide ships a minimal Cloud Run service that serves the web UI plus JSON endpoints. It relies
on the `/api/meteogram/run` endpoint to refresh data, so you can wire Cloud Scheduler to call it on
a cadence.

## Prerequisites
- `gcloud` installed and authenticated (`gcloud auth login`)
- A Google Cloud project with billing enabled

## 1) Create and configure the project
```bash
export PROJECT_ID="wxgraph-<your-suffix>"
export REGION="us-central1"

gcloud projects create "${PROJECT_ID}"
gcloud config set project "${PROJECT_ID}"
gcloud billing projects link "${PROJECT_ID}" --billing-account "<billing-account-id>"

gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  cloudbuild.googleapis.com scheduler.googleapis.com
```

## 2) Create a storage bucket (recommended)
```bash
export BUCKET_NAME="wxgraph-latest-${PROJECT_ID}"
gcloud storage buckets create "gs://${BUCKET_NAME}" --location "${REGION}"
```

## 3) Create a dedicated service account (recommended)
```bash
export SERVICE_ACCOUNT="wxgraph-runner"
gcloud iam service-accounts create "${SERVICE_ACCOUNT}" \
  --display-name "wxGraph Cloud Run"

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member "serviceAccount:${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role "roles/storage.objectAdmin"
```

## 4) Build and deploy to Cloud Run
```bash
export SERVICE_NAME="wxgraph"

gcloud run deploy "${SERVICE_NAME}" \
  --source . \
  --region "${REGION}" \
  --allow-unauthenticated \
  --service-account "${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --set-env-vars WXGRAPH_LAT=41.48,WXGRAPH_LON=-81.81,WXGRAPH_SITE=KCLE,WXGRAPH_GCS_BUCKET="${BUCKET_NAME}",WXGRAPH_GCS_PREFIX=latest \
  --min-instances 1 \
  --no-cpu-throttling
```

`--min-instances 1` keeps one container warm so `/api/meteogram/run` writes to a stable, in-memory
filesystem. If you let the service scale to zero, the latest JSON/PNG will be rebuilt on the next
run or request.

## 5) Seed the initial meteogram
```bash
export SERVICE_URL="$(gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --format='value(status.url)')"
curl -X POST "${SERVICE_URL}/api/meteogram/run"
```

## 6) Automate refreshes (optional but recommended)
```bash
gcloud scheduler jobs create http wxgraph-refresh \
  --schedule "0 */2 * * *" \
  --http-method POST \
  --uri "${SERVICE_URL}/api/meteogram/run" \
  --time-zone "UTC"
```

If you set `WXGRAPH_RUN_TOKEN`, pass it via Scheduler headers:
```bash
gcloud scheduler jobs update http wxgraph-refresh \
  --location "${REGION}" \
  --headers "x-run-token=YOUR_TOKEN"
```

## Endpoints
- Web UI: `${SERVICE_URL}/`
- Latest JSON: `${SERVICE_URL}/api/meteogram/latest`
- Health: `${SERVICE_URL}/api/health`
- Trigger refresh: `POST ${SERVICE_URL}/api/meteogram/run`

## Notes on storage
Cloud Run filesystems are ephemeral. When `WXGRAPH_GCS_BUCKET` is set the service downloads the
latest meteogram JSON/PNG from Cloud Storage on demand and uploads fresh results after each run.
This keeps data available across restarts and cold starts.
