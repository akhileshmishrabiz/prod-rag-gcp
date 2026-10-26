# Deploy and operate on GCP

## What Terraform creates

- Artifact Registry repository for API and worker images
- GCS bucket for source documents
- Firestore (Native) database
- Pub/Sub topic, push subscription (OIDC to worker), and DLQ topic
- Vertex AI Vector Search index (stream updates) and private endpoint
- Cloud Run API (public ingress, IAM invokers only) and worker (internal ingress)
- Service accounts and least-privilege IAM

## Worker audience bootstrap

Cloud Run sets `PUBSUB_WORKER_AUDIENCE` to the worker service URL. After the first deploy, if the worker fails readiness because the audience was empty, run `terraform apply` again once the worker URI exists.

## IAM

| Identity | Can |
|----------|-----|
| Your user (via `api_invokers`) | Invoke API |
| API service account | Read GCS objects, read/write Firestore, publish Pub/Sub, call Vertex |
| Worker service account | Read GCS, read/write Firestore, upsert/query Vector Search |
| Pub/Sub push SA | `run.invoker` on worker only |

## Browser UI and 403 Forbidden

Cloud Run enforces IAM on the URL itself. Console sign-in does not send an ID token with normal navigation. If `api_allow_unauthenticated` is false and org policy blocks public invokers, use:

```bash
./scripts/open-ui.sh
```

Then open `http://127.0.0.1:8080`.

## Troubleshooting

```bash
gcloud run services logs read rag-main-api --region=asia-south1
gcloud run services logs read rag-main-worker --region=asia-south1
curl "$API_URL/readyz" -H "Authorization: Bearer $(gcloud auth print-identity-token)"
```

- **403 in browser on the Cloud Run URL**: expected without a bearer token or IAP/LB; use `scripts/open-ui.sh` or curl with `gcloud auth print-identity-token`.
- **503 on `/readyz`**: missing env, Firestore unreachable, or active-index mismatch.
- **Document stuck PENDING**: check worker logs and Pub/Sub subscription metrics.
- **Empty answers**: confirm document `READY`, embeddings model region matches `GCP_REGION`.
