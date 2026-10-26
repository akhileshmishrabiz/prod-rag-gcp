#!/bin/sh
set -eu

fail() {
  printf 'local startup rejected: %s\n' "$1" >&2
  exit 78
}

[ "${APP_MODE:-}" = "local" ] || fail "APP_MODE must be local"
[ "${AUTH_MODE:-}" = "local_signed" ] || fail "AUTH_MODE must be local_signed"
[ "${GOOGLE_CLOUD_PROJECT:-}" = "local-rag-project" ] || fail "emulator project is required"
[ -n "${LOCAL_AUTH_SIGNING_KEY_FILE:-}" ] || fail "local signing key path is required"
[ -r "${LOCAL_AUTH_SIGNING_KEY_FILE}" ] || fail "local signing key is not readable"
[ -n "${LOCAL_METADATA_ENDPOINT:-}" ] || fail "metadata emulator endpoint is required"
[ -n "${LOCAL_GCS_ENDPOINT:-}" ] || fail "storage emulator endpoint is required"
[ -n "${LOCAL_PUBSUB_ENDPOINT:-}" ] || fail "Pub/Sub emulator endpoint is required"
[ -n "${LOCAL_EMBEDDING_ENDPOINT:-}" ] || fail "embedding emulator endpoint is required"
[ -n "${LOCAL_VECTOR_ENDPOINT:-}" ] || fail "vector emulator endpoint is required"
[ -n "${LOCAL_GENERATION_ENDPOINT:-}" ] || fail "generation emulator endpoint is required"

case "${LOCAL_METADATA_ENDPOINT} ${LOCAL_GCS_ENDPOINT} ${LOCAL_PUBSUB_ENDPOINT} ${LOCAL_EMBEDDING_ENDPOINT} ${LOCAL_VECTOR_ENDPOINT} ${LOCAL_GENERATION_ENDPOINT}" in
  *googleapis.com*|*google.com*|*gcr.io*) fail "public Google endpoints are forbidden" ;;
esac

[ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ] || fail "cloud credential files are forbidden"
[ -z "${CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE:-}" ] || fail "Cloud SDK credentials are forbidden"
[ "${HTTP_PROXY:-}" = "http://127.0.0.1:9" ] || fail "blocked egress proxy is required"
[ "${HTTPS_PROXY:-}" = "http://127.0.0.1:9" ] || fail "blocked TLS egress proxy is required"

exec "$@"
