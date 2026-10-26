variable "project_id" {
  type        = string
  description = "GCP project ID"
}

variable "region" {
  type        = string
  description = "Single region for all services (e.g. us-central1)"
  default     = "us-central1"
}

variable "tenant_id" {
  type        = string
  description = "Fixed tenant identifier for this deployment"
  default     = "main"
}

variable "embedding_dimensions" {
  type        = number
  description = "Must match EMBEDDING_DIMENSIONS on Cloud Run"
  default     = 768
}

variable "generation_model_id" {
  type    = string
  default = "gemini-2.5-flash-001"
}

variable "embedding_model_id" {
  type    = string
  default = "text-embedding-005"
}

variable "api_image" {
  type        = string
  description = "Container image for the API (Artifact Registry or gcr.io)"
}

variable "worker_image" {
  type        = string
  description = "Container image for the ingestion worker"
}

variable "api_invokers" {
  type        = list(string)
  description = "IAM members allowed to call the API (user: or group:)"
  default     = []
}

variable "api_allow_unauthenticated" {
  type        = bool
  description = "Allow public browser access to the API and UI (demo only)"
  default     = false
}
