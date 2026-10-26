# Codebase guide

## Layout

```text
services/api/          HTTP API
services/worker/       Pub/Sub ingestion
services/composition/  GCP dependency wiring at startup
packages/rag_core/     Domain logic and Vertex adapters
terraform/             Infrastructure
contracts/             OpenAPI and config schema
tests/                 Unit tests with fakes
```

## Request flows

**Register document:** API validates GCS object → Firestore `PENDING` → Pub/Sub event → worker ingests → `READY`.

**Query:** API embeds question → Vector Search with tenant/corpus/generation filters → Firestore hydration → Gemini with evidence only.

## Changing behavior

| Goal | Start here |
|------|------------|
| API endpoint | `services/api/app/routes/` |
| Ingestion | `services/worker/app/service.py` |
| Retrieval policy | `packages/rag_core/retrieval.py` |
| GCP wiring | `services/composition/gcp.py` |
| Infrastructure | `terraform/` |
