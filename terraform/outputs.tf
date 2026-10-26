output "api_url" {
  value       = google_cloud_run_v2_service.api.uri
  description = "Cloud Run URL for the RAG API"
}

output "document_bucket" {
  value       = google_storage_bucket.documents.name
  description = "Upload documents under gs://BUCKET/TENANT_UPLOAD_PREFIX/"
}

output "upload_prefix" {
  value       = local.upload_prefix
  description = "Object prefix for tenant uploads"
}

output "artifact_registry" {
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
  description = "Docker repository for API and worker images"
}
