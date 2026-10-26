"""Fail-closed tenant, object, and vector restriction policy."""

from __future__ import annotations

import hmac
from urllib.parse import unquote, urlsplit

from packages.rag_core.errors import ErrorCode, RagError
from packages.rag_core.models import (
    Corpus,
    DocumentId,
    ObjectMetadata,
    VectorDatapoint,
    VectorRestrictions,
)


def require_tenant(actual: str, configured: str) -> None:
    """Reject absent or non-configured tenants without leaking values."""
    if not actual or not hmac.compare_digest(actual, configured):
        raise RagError(ErrorCode.TENANT_MISMATCH)


def active_generation_token(document_id: str, generation: int) -> str:
    """Create an unambiguous generation token used by every vector operation."""
    if not document_id or generation < 1:
        raise RagError(ErrorCode.INVALID_REQUEST)
    return f"{len(document_id)}:{document_id}:{generation}"


def build_search_restrictions(
    *,
    tenant_id: str,
    corpus: str,
    index_version: str,
    active_generations: list[tuple[str, int]],
) -> VectorRestrictions:
    """Build mandatory exact-match restrictions before a vector call."""
    if not tenant_id or not corpus or not index_version or not active_generations:
        raise RagError(ErrorCode.INVALID_REQUEST)
    tokens = tuple(
        sorted(
            {
                active_generation_token(document_id, generation)
                for document_id, generation in active_generations
            }
        )
    )
    if not tokens:
        raise RagError(ErrorCode.INVALID_REQUEST)
    return VectorRestrictions(
        tenant_id=tenant_id,
        corpus=corpus,
        index_version=index_version,
        active_generations=tokens,
    )


def build_datapoint(
    *,
    vector_id: str,
    vector: list[float],
    tenant_id: str,
    corpus: str,
    document_id: str,
    generation: int,
    index_version: str,
) -> VectorDatapoint:
    """Attach all mandatory restrict namespaces to an upsert."""
    if not all((vector_id, vector, tenant_id, corpus, document_id, index_version)):
        raise RagError(ErrorCode.INVALID_REQUEST)
    return VectorDatapoint(
        vector_id=vector_id,
        vector=vector,
        tenant_id=tenant_id,
        corpus=corpus,
        document_id=document_id,
        generation=generation,
        active_generation=active_generation_token(document_id, generation),
        index_version=index_version,
    )


class GcsObjectPolicy:
    """Validate canonical objects against one managed bucket and tenant prefix."""

    def __init__(
        self,
        *,
        bucket: str,
        tenant_id: str,
        max_bytes: int,
        allowed_media_types: frozenset[str] | None = None,
    ) -> None:
        if not bucket or not tenant_id or max_bytes < 1:
            raise ValueError("invalid GCS policy configuration")
        self.bucket = bucket
        self.prefix = f"tenants/{tenant_id}/"
        self.max_bytes = max_bytes
        self.allowed_media_types = allowed_media_types or frozenset(
            {"text/plain", "text/markdown", "application/pdf"}
        )

    def parse_uri(self, uri: str) -> tuple[str, str]:
        parts = urlsplit(uri)
        name = unquote(parts.path.lstrip("/"))
        if (
            parts.scheme != "gs"
            or parts.netloc != self.bucket
            or parts.query
            or parts.fragment
            or not name.startswith(self.prefix)
            or name.endswith("/")
            or ".." in name.split("/")
        ):
            raise RagError(ErrorCode.ACCESS_DENIED)
        return parts.netloc, name

    def validate(
        self,
        metadata: ObjectMetadata,
        *,
        expected_uri: str,
        expected_object_generation: int,
        expected_checksum: str | None = None,
    ) -> None:
        bucket, name = self.parse_uri(expected_uri)
        if metadata.bucket != bucket or metadata.name != name:
            raise RagError(ErrorCode.DEPENDENCY_REJECTED)
        if (
            metadata.canonical_uri != expected_uri
            or metadata.object_generation != expected_object_generation
        ):
            raise RagError(ErrorCode.STALE_GENERATION)
        if metadata.size > self.max_bytes:
            raise RagError(ErrorCode.DOCUMENT_TOO_LARGE)
        if metadata.content_type.value not in self.allowed_media_types:
            raise RagError(ErrorCode.UNSUPPORTED_MEDIA_TYPE)
        if expected_checksum is not None and metadata.checksum_sha256 != expected_checksum:
            raise RagError(ErrorCode.DEPENDENCY_REJECTED)


def validate_scope(tenant_id: str, corpus: Corpus, document_id: DocumentId) -> None:
    """Static typing helper that also rejects empty runtime values."""
    if not tenant_id or not corpus or not document_id:
        raise RagError(ErrorCode.INVALID_REQUEST)
