# GCP RAG

Grounded retrieval-augmented generation on Google Cloud: FastAPI on Cloud Run, Firestore metadata, GCS documents, Pub/Sub ingestion, Vertex embeddings, Vector Search, and Gemini.

## Architecture

```text
You → Cloud Run API → GCS / Firestore / Pub/Sub / Vertex
Pub/Sub → Cloud Run worker → parse, chunk, embed, index → Vector Search
Query → embed → Vector Search → hydrate Firestore → Gemini → cited answer
```

## Prerequisites

- Google Cloud project with billing enabled
- `gcloud`, `docker`, `terraform` (≥ 1.5), and [`uv`](https://docs.astral.sh/uv/)
- Python 3.12 for running tests locally

## Deploy

1. Authenticate:

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

2. Copy Terraform variables and add your email for API access:

```bash
cp terraform/terraform.tfvars.example terraform/terraform.tfvars
# edit project_id, api_invokers, etc.
```

3. Build, push, and apply (or run the helper script):

```bash
export PROJECT_ID=YOUR_PROJECT_ID
export REGION=us-central1
chmod +x scripts/deploy.sh
./scripts/deploy.sh
```

Or manually: enable APIs via first `terraform apply`, push images to the Artifact Registry repo Terraform creates, then pass `api_image` and `worker_image` in `terraform.tfvars`.

4. Note outputs:

```bash
cd terraform && terraform output api_url document_bucket upload_prefix
```

Grant yourself `roles/run.invoker` on the API if not listed in `api_invokers`.

## Web UI in the browser

The API serves a small UI at `/`. Cloud Run checks **IAM before your app runs**: every request needs a Google **ID token** in `Authorization: Bearer …`, or the service must allow unauthenticated invoke.

Being signed into Google Cloud Console does **not** attach that token to a normal tab visit, so opening the Cloud Run URL directly often returns **403 Forbidden** even when your user has `roles/run.invoker`.

Many organizations also block binding `allUsers` or `allAuthenticatedUsers` on Cloud Run, so “make it public” may fail with an org-policy error.

**Workaround (recommended for now):** run a local proxy that injects your `gcloud` identity token, then open localhost:

```bash
gcloud auth login   # same Google account as in api_invokers
chmod +x scripts/open-ui.sh
./scripts/open-ui.sh
# open http://127.0.0.1:8080
```

For a hosted sign-in page on the public Cloud Run URL, you typically need **HTTPS load balancing + Identity-Aware Proxy** (or an org-policy exception to allow unauthenticated invoke on this service).

## Use the API

1. Upload a text or PDF file to  
   `gs://DOCUMENT_BUCKET/tenants/TENANT_ID/uploads/my-doc.txt`  
   and set object metadata `sha256` to the file’s SHA-256 hex digest.

2. Get an identity token and register the document:

```bash
API_URL="$(cd terraform && terraform output -raw api_url)"
TOKEN="$(gcloud auth print-identity-token)"
curl -sS -X POST "$API_URL/v1/documents" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "my-doc",
    "object_uri": "gs://BUCKET/tenants/TENANT/uploads/my-doc.txt",
    "object_generation": GENERATION,
    "checksum_sha256": "SHA256",
    "media_type": "text/plain",
    "idempotency_key": "register-my-doc-1"
  }'
```

3. Poll `GET /v1/documents/my-doc` until `status` is `READY`.

4. Ask a question:

```bash
curl -sS -X POST "$API_URL/v1/query" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the document say about …?"}'
```

## Development

```bash
uv python install 3.12
uv sync --frozen --all-groups
make check
```

Unit tests use in-memory fakes; they do not call Google Cloud.

## Repository layout

| Path | Purpose |
|------|---------|
| `services/api` | HTTP API |
| `services/worker` | Pub/Sub ingestion worker |
| `services/composition` | Wires GCP clients at startup |
| `packages/rag_core` | RAG domain logic and Vertex adapters |
| `terraform/` | Infrastructure |
| `contracts/` | OpenAPI and config schema |

## Cost note

Vector Search runs a small deployed index (`SHARD_SIZE_SMALL`) with non-zero hourly cost even when Cloud Run scales to zero. Tear down with `terraform destroy` when not in use.

See [`docs/deploy.md`](docs/deploy.md) for IAM details and troubleshooting.
