#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TF_DIR="$ROOT/terraform"

: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:=us-central1}"
: "${TENANT_ID:=main}"

REPO="${REGION}-docker.pkg.dev/${PROJECT_ID}/rag-${TENANT_ID}-images"
API_IMAGE="${REPO}/api:$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo latest)"
WORKER_IMAGE="${REPO}/worker:$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo latest)"

gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

docker build -f "$ROOT/Dockerfile.api" -t "$API_IMAGE" "$ROOT"
docker build -f "$ROOT/Dockerfile.worker" -t "$WORKER_IMAGE" "$ROOT"
docker push "$API_IMAGE"
docker push "$WORKER_IMAGE"

cd "$TF_DIR"
terraform init
terraform apply \
  -var="project_id=${PROJECT_ID}" \
  -var="region=${REGION}" \
  -var="tenant_id=${TENANT_ID}" \
  -var="api_image=${API_IMAGE}" \
  -var="worker_image=${WORKER_IMAGE}"

terraform output
