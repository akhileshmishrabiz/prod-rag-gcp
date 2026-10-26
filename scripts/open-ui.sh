#!/usr/bin/env bash
# Local proxy so the browser can use the RAG UI without sending Cloud Run ID tokens.
# Cloud Run IAM is enforced before your app; Google account cookies from the Console do not count.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TFVARS="${ROOT}/terraform/terraform.tfvars"

PROJECT="${PROJECT_ID:-}"
REGION="${REGION:-us-central1}"
SERVICE="${API_SERVICE:-rag-main-api}"
PORT="${PORT:-8080}"

if [[ -f "${TFVARS}" ]]; then
  # shellcheck disable=SC1090
  while IFS= read -r line; do
    [[ "${line}" =~ ^project_id[[:space:]]*= ]] && PROJECT="${line#*=}" && PROJECT="${PROJECT//\"/}" && PROJECT="${PROJECT// /}"
    [[ "${line}" =~ ^region[[:space:]]*= ]] && REGION="${line#*=}" && REGION="${REGION//\"/}" && REGION="${REGION// /}"
  done < "${TFVARS}"
fi

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud not found. Install the Google Cloud SDK and run: gcloud auth login" >&2
  exit 1
fi

if [[ -z "${PROJECT}" ]]; then
  PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
fi
if [[ -z "${PROJECT}" ]]; then
  echo "Set PROJECT_ID or project_id in terraform/terraform.tfvars, or: gcloud config set project YOUR_PROJECT" >&2
  exit 1
fi

echo "Using project=${PROJECT} region=${REGION} service=${SERVICE}"
echo "Open http://127.0.0.1:${PORT} (this proxy attaches your gcloud identity token)."
echo "Press Ctrl+C to stop."
exec gcloud run services proxy "${SERVICE}" --region="${REGION}" --project="${PROJECT}" --port="${PORT}"
