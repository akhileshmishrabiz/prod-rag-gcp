# Local-First GCP RAG

A fixed-tenant RAG development service built with Python 3.12, FastAPI, and
Google Cloud adapters. Docker Compose runs the API, ingestion worker, Firestore
and Pub/Sub emulators, fake GCS, and a deterministic model/vector service
without cloud credentials or billable calls.

## Run locally

Prerequisites: Docker Compose v2 and
[`uv`](https://docs.astral.sh/uv/).

```bash
uv python install 3.12
uv sync --frozen --all-groups
make check
make local-run
```

The API is available at `http://127.0.0.1:8080`. Stop and remove local data
with:

```bash
make local-down
```

See [`docs/local-development.md`](docs/local-development.md) for services,
architecture, commands, and troubleshooting.

New contributors should start with
[`docs/codebase-guide.md`](docs/codebase-guide.md) for the directory layout,
request flows, dependency boundaries, tests, and common change locations.

Useful individual commands:

```bash
docker compose -f local/compose.yaml up --build --wait
python3 local/scripts/seed.py
python3 local/scripts/smoke.py
make format
make check
```

## Repository layout

- `services/api`: document and grounded-query HTTP API
- `services/worker`: authenticated Pub/Sub ingestion endpoint
- `packages/rag_core`: domain models, policies, adapters, parsing, chunking, and retrieval
- `contracts`: OpenAPI and runtime configuration schemas
- `tests`: production application and adapter tests
- `local`: isolated Compose topology, emulators, local wiring, scripts, and tests
- `docs/architecture.md`: architecture, trust boundaries, and known limitations

The intended GCP services, architecture, configuration, and missing deployment
work are documented in
[`docs/production-gcp.md`](docs/production-gcp.md).

## Safety

Local mode uses explicit emulator endpoints and signed development identities.
It rejects public Google endpoints and does not mount Application Default
Credentials. GCP mode is fail-closed: it requires Cloud Run IAM, resolved
regional configuration, pinned model IDs, and no local adapters.

This is a single-region development system, not a production, HA, DR,
compliance, or arbitrary-multitenant design. No Terraform or live GCP
deployment is currently included.