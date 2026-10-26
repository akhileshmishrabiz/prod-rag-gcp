locals {
  name_prefix   = "rag-${var.tenant_id}"
  upload_prefix = "tenants/${var.tenant_id}/uploads"
  deployed_id   = "rag_deployed"
  runtime_env = {
    CONFIG_CONTRACT_VERSION         = "1.0.0"
    DEV_TENANT_ID                   = var.tenant_id
    GCP_PROJECT_ID                  = var.project_id
    GCP_REGION                      = var.region
    GCS_DOCUMENT_BUCKET             = google_storage_bucket.documents.name
    TENANT_UPLOAD_PREFIX            = local.upload_prefix
    FIXED_CORPUS                    = "default"
    PUBSUB_TOPIC                    = google_pubsub_topic.ingestion.name
    ACTIVE_INDEX_VERSION            = "index-1"
    PIPELINE_VERSION                = "pipeline-1"
    PARSER_VERSION                  = "parser-1"
    CHUNKER_VERSION                 = "chunker-1"
    GENERATION_MODEL_ID             = var.generation_model_id
    EMBEDDING_MODEL_ID              = var.embedding_model_id
    EMBEDDING_DIMENSIONS            = tostring(var.embedding_dimensions)
    MAX_DOCUMENT_BYTES              = "10485760"
    MAX_DOCUMENT_CHUNKS             = "1000"
    MAX_EMBEDDING_BATCH_SIZE        = "100"
    MAX_QUERY_INPUT_CHARS           = "4000"
    MAX_QUERY_TOP_K                 = "20"
    MAX_ACTIVE_GENERATIONS          = "1000"
    MAX_CANDIDATES_PER_DOCUMENT     = "5"
    MAX_CONTEXT_CHARS               = "40000"
    MAX_CONTEXT_TOKENS              = "8000"
    MAX_EVIDENCE_CHARS              = "6000"
    MAX_OUTPUT_TOKENS               = "1024"
    MAX_PDF_PAGES                   = "200"
    CHUNK_SIZE_TOKENS               = "512"
    CHUNK_OVERLAP_TOKENS            = "64"
    DEPENDENCY_TIMEOUT_SECONDS      = "30"
    PROCESSING_DEADLINE_SECONDS     = "600"
    PROCESSING_LEASE_SECONDS        = "300"
    MAX_PROCESSING_ATTEMPTS         = "5"
    EMBEDDING_MAX_RETRIES           = "3"
    WORKER_CONCURRENCY              = "4"
    API_MAX_INSTANCES               = "2"
    WORKER_MAX_INSTANCES            = "2"
    VECTOR_SEARCH_INDEX_RESOURCE    = google_vertex_ai_index.vectors.id
    VECTOR_SEARCH_ENDPOINT_RESOURCE = google_vertex_ai_index_endpoint.vectors.id
    VECTOR_SEARCH_DEPLOYED_INDEX_ID = local.deployed_id
  }
}

resource "google_project_service" "required" {
  for_each = toset([
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "storage.googleapis.com",
    "firestore.googleapis.com",
    "pubsub.googleapis.com",
    "aiplatform.googleapis.com",
    "iam.googleapis.com",
  ])
  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

resource "google_artifact_registry_repository" "images" {
  location      = var.region
  repository_id = "${local.name_prefix}-images"
  format        = "DOCKER"
  depends_on    = [google_project_service.required]
}

resource "google_storage_bucket" "documents" {
  name                        = "${local.name_prefix}-docs-${var.project_id}"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = true
  depends_on                  = [google_project_service.required]
}

resource "google_firestore_database" "default" {
  project     = var.project_id
  name        = "(default)"
  location_id = var.region
  type        = "FIRESTORE_NATIVE"
  depends_on  = [google_project_service.required]
}

resource "google_pubsub_topic" "ingestion" {
  name       = "ingestion"
  depends_on = [google_project_service.required]
}

resource "google_pubsub_topic" "dlq" {
  name       = "ingestion-dlq"
  depends_on = [google_project_service.required]
}

resource "google_storage_bucket" "vector_staging" {
  name                        = "${local.name_prefix}-vector-${var.project_id}"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = true
  depends_on                  = [google_project_service.required]
}

resource "google_vertex_ai_index" "vectors" {
  display_name = "${local.name_prefix}-index"
  region       = var.region
  description  = "RAG chunk vectors"
  metadata {
    contents_delta_uri = "gs://${google_storage_bucket.vector_staging.name}/initial"
    config {
      dimensions                  = var.embedding_dimensions
      approximate_neighbors_count = 20
      distance_measure_type       = "DOT_PRODUCT_DISTANCE"
      shard_size                  = "SHARD_SIZE_SMALL"
      algorithm_config {
        tree_ah_config {
          leaf_node_embedding_count    = 500
          leaf_nodes_to_search_percent = 7
        }
      }
    }
  }
  index_update_method = "STREAM_UPDATE"
  depends_on          = [google_project_service.required]
}

resource "google_vertex_ai_index_endpoint" "vectors" {
  display_name            = "${local.name_prefix}-endpoint"
  region                  = var.region
  public_endpoint_enabled = true
  depends_on              = [google_vertex_ai_index.vectors]
}

resource "google_vertex_ai_index_endpoint_deployed_index" "vectors" {
  index_endpoint    = google_vertex_ai_index_endpoint.vectors.id
  index             = google_vertex_ai_index.vectors.id
  deployed_index_id = local.deployed_id
  dedicated_resources {
    machine_spec {
      machine_type = "e2-standard-2"
    }
    min_replica_count = 1
    max_replica_count = 1
  }
  depends_on = [google_vertex_ai_index_endpoint.vectors]
}

resource "google_service_account" "api" {
  account_id   = "${local.name_prefix}-api"
  display_name = "RAG API runtime"
}

resource "google_service_account" "worker" {
  account_id   = "${local.name_prefix}-worker"
  display_name = "RAG ingestion worker"
}

resource "google_service_account" "pubsub_push" {
  account_id   = "${local.name_prefix}-pubsub"
  display_name = "Pub/Sub push to worker"
}

resource "google_project_iam_member" "api_roles" {
  for_each = toset([
    "roles/aiplatform.user",
    "roles/datastore.user",
    "roles/pubsub.publisher",
    "roles/storage.objectCreator",
  ])
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.api.email}"
}

resource "google_storage_bucket_iam_member" "api_objects" {
  bucket = google_storage_bucket.documents.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.api.email}"
}

resource "google_project_iam_member" "worker_roles" {
  for_each = toset([
    "roles/aiplatform.user",
    "roles/datastore.user",
  ])
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.worker.email}"
}

resource "google_storage_bucket_iam_member" "worker_objects" {
  bucket = google_storage_bucket.documents.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.worker.email}"
}

resource "google_cloud_run_v2_service" "worker" {
  name                = "${local.name_prefix}-worker"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false

  template {
    service_account = google_service_account.worker.email
    containers {
      image = var.worker_image
      ports {
        container_port = 8081
      }
      dynamic "env" {
        for_each = merge(local.runtime_env, {
          # Worker does not use this at runtime; HTTPS placeholder satisfies config validation.
          PUBSUB_WORKER_AUDIENCE = "https://worker-audience.invalid"
        })
        content {
          name  = env.key
          value = env.value
        }
      }
      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
      }
    }
    scaling {
      max_instance_count = 2
    }
  }

  depends_on = [
    google_vertex_ai_index_endpoint_deployed_index.vectors,
    google_firestore_database.default,
  ]
}

resource "google_cloud_run_v2_service" "api" {
  name                = "${local.name_prefix}-api"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false

  template {
    service_account = google_service_account.api.email
    containers {
      image = var.api_image
      ports {
        container_port = 8080
      }
      dynamic "env" {
        for_each = merge(local.runtime_env, {
          PUBSUB_WORKER_AUDIENCE = google_cloud_run_v2_service.worker.uri
        })
        content {
          name  = env.key
          value = env.value
        }
      }
      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
      }
    }
    scaling {
      max_instance_count = 2
    }
  }

  depends_on = [google_cloud_run_v2_service.worker]
}

resource "google_cloud_run_v2_service_iam_member" "worker_invoker_pubsub" {
  name     = google_cloud_run_v2_service.worker.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.pubsub_push.email}"
}

resource "google_pubsub_subscription" "worker" {
  name  = "ingestion-worker"
  topic = google_pubsub_topic.ingestion.name

  push_config {
    push_endpoint = "${google_cloud_run_v2_service.worker.uri}/v1/pubsub/push"
    oidc_token {
      service_account_email = google_service_account.pubsub_push.email
      audience              = google_cloud_run_v2_service.worker.uri
    }
  }

  dead_letter_policy {
    dead_letter_topic     = google_pubsub_topic.dlq.id
    max_delivery_attempts = 5
  }

  ack_deadline_seconds = 600
  depends_on           = [google_cloud_run_v2_service_iam_member.worker_invoker_pubsub]
}

resource "google_cloud_run_v2_service_iam_member" "api_invokers" {
  for_each = toset(var.api_invokers)
  name     = google_cloud_run_v2_service.api.name
  location = var.region
  role     = "roles/run.invoker"
  member   = each.value
}

resource "google_cloud_run_v2_service_iam_member" "api_public" {
  count    = var.api_allow_unauthenticated ? 1 : 0
  name     = google_cloud_run_v2_service.api.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}
