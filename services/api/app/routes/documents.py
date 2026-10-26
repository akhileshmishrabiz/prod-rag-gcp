"""Document registration, inspection, and tombstone-first deletion."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Request, status

from packages.rag_core.errors import ErrorCode, RagError
from packages.rag_core.models import (
    DocumentRecord,
    DocumentRegistration,
    IngestionMessage,
    IngestionOperation,
    MessageMetadata,
    RegistrationOutcome,
)
from packages.rag_core.policy import GcsObjectPolicy
from services.api.app.auth import require_identity
from services.api.app.dependencies import ApiDependencies
from services.api.app.schemas import DocumentResponse, RegisterDocumentRequest

router = APIRouter(
    prefix="/v1/documents",
    tags=["documents"],
    dependencies=[Depends(require_identity)],
)


def _dependencies(request: Request) -> ApiDependencies:
    dependencies: ApiDependencies | None = request.app.state.dependencies
    if dependencies is None:
        raise RagError(ErrorCode.CONFIGURATION_INVALID)
    return dependencies


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=DocumentResponse,
    operation_id="registerDocument",
)
def register_document(body: RegisterDocumentRequest, request: Request) -> DocumentResponse:
    settings = request.app.state.settings
    dependencies = _dependencies(request)
    if settings is None:
        raise RagError(ErrorCode.CONFIGURATION_INVALID)
    if body.corpus != settings.fixed_corpus:
        raise RagError(ErrorCode.ACCESS_DENIED)
    object_policy = GcsObjectPolicy(
        bucket=settings.gcs_document_bucket,
        tenant_id=settings.dev_tenant_id,
        max_bytes=settings.max_document_bytes,
    )
    _, object_name = object_policy.parse_uri(body.object_uri)
    if not object_name.startswith(f"{settings.tenant_upload_prefix}/"):
        raise RagError(ErrorCode.ACCESS_DENIED)
    metadata = dependencies.objects.stat(body.object_uri, settings.dependency_timeout_seconds)
    object_policy.validate(
        metadata,
        expected_uri=body.object_uri,
        expected_object_generation=body.object_generation,
        expected_checksum=body.checksum_sha256,
    )
    if metadata.content_type is not body.media_type:
        raise RagError(ErrorCode.UNSUPPORTED_MEDIA_TYPE)

    now = dependencies.clock.now()
    result = dependencies.metadata.register(
        DocumentRegistration(
            tenant_id=settings.dev_tenant_id,
            document_id=body.document_id,
            corpus=body.corpus,
            object_uri=body.object_uri,
            object_generation=body.object_generation,
            checksum_sha256=body.checksum_sha256,
            media_type=body.media_type,
            pipeline_version=settings.pipeline_version,
        ),
        body.idempotency_key,
        dependencies.ids.new_event_id(),
        now,
    )
    if result.idempotency.outcome is RegistrationOutcome.COMMITTED:
        document = result.document
        if document is None:
            raise RagError(ErrorCode.DOCUMENT_CONFLICT)
        message = _message(
            request=request,
            operation=IngestionOperation.UPSERT,
            document=document,
            event_id=result.idempotency.event_id,
        )
        dependencies.publisher.publish(message, settings.dependency_timeout_seconds)
        dependencies.metadata.mark_registration_published(
            settings.dev_tenant_id,
            result.idempotency.idempotency_key_sha256,
            result.idempotency.request_sha256,
            result.idempotency.event_id,
            dependencies.clock.now(),
        )
    document = result.document or dependencies.metadata.get_document(
        settings.dev_tenant_id, result.idempotency.document_id
    )
    if document is None or document.generation != result.idempotency.generation:
        raise RagError(ErrorCode.DOCUMENT_CONFLICT)
    return DocumentResponse.from_record(document)


@router.get("/{document_id}", response_model=DocumentResponse, operation_id="getDocument")
def get_document(document_id: str, request: Request) -> DocumentResponse:
    settings = request.app.state.settings
    dependencies = _dependencies(request)
    if settings is None:
        raise RagError(ErrorCode.CONFIGURATION_INVALID)
    document = dependencies.metadata.get_document(settings.dev_tenant_id, document_id)
    if document is None:
        raise RagError(ErrorCode.DOCUMENT_NOT_FOUND)
    return DocumentResponse.from_record(document)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=DocumentResponse,
    operation_id="deleteDocument",
)
def delete_document(document_id: str, request: Request) -> DocumentResponse:
    settings = request.app.state.settings
    dependencies = _dependencies(request)
    if settings is None:
        raise RagError(ErrorCode.CONFIGURATION_INVALID)
    document = dependencies.metadata.tombstone(
        settings.dev_tenant_id, document_id, dependencies.clock.now()
    )
    # Tombstone is already committed. Publication failure is surfaced as
    # retryable while hydration remains blocked immediately.
    message = _message(
        request=request,
        operation=IngestionOperation.DELETE,
        document=document,
        event_id=dependencies.ids.new_event_id(),
    )
    dependencies.publisher.publish(message, settings.dependency_timeout_seconds)
    return DocumentResponse.from_record(document)


def _message(
    *,
    request: Request,
    operation: IngestionOperation,
    document: DocumentRecord,
    event_id: UUID,
) -> IngestionMessage:
    trace_id = request.state.trace_id
    return IngestionMessage(
        operation=operation,
        tenant_id=document.tenant_id,
        document_id=document.document_id,
        object_uri=document.object_uri,
        object_generation=document.object_generation,
        generation=document.generation,
        checksum_sha256=document.checksum_sha256,
        corpus=document.corpus,
        pipeline_version=document.pipeline_version,
        trace_id=trace_id,
        message_metadata=MessageMetadata(
            event_id=event_id,
            occurred_at=request.app.state.dependencies.clock.now(),
        ),
    )
