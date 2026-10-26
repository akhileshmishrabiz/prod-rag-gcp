"""Document registration, inspection, and tombstone-first deletion."""

from __future__ import annotations

import hashlib
import re
from pathlib import PurePath
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile, status

from packages.rag_core.errors import ErrorCode, RagError
from packages.rag_core.models import (
    DocumentRecord,
    DocumentRegistration,
    IngestionMessage,
    IngestionOperation,
    MediaType,
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

_DOCUMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_INVALID_DOC_ID_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _resolve_document_id(filename: str, document_id: str | None) -> str:
    if document_id is not None:
        candidate = document_id.strip()
    else:
        stem = PurePath(filename).stem
        candidate = _INVALID_DOC_ID_CHARS.sub("-", stem).strip("-")
        candidate = re.sub(r"-{2,}", "-", candidate)
        if not candidate:
            candidate = "document"
        if not candidate[0].isalnum():
            candidate = f"doc-{candidate.lstrip('._-')}" or "document"
    if len(candidate) > 128:
        candidate = candidate[:128].rstrip("._-")
        if not candidate or not candidate[0].isalnum():
            candidate = f"doc-{candidate}"[:128]
    if not _DOCUMENT_ID.fullmatch(candidate):
        raise RagError(ErrorCode.INVALID_REQUEST)
    return candidate


_MEDIA_BY_SUFFIX = {
    ".txt": MediaType.TEXT,
    ".md": MediaType.MARKDOWN,
    ".markdown": MediaType.MARKDOWN,
    ".pdf": MediaType.PDF,
}


def _dependencies(request: Request) -> ApiDependencies:
    dependencies: ApiDependencies | None = request.app.state.dependencies
    if dependencies is None:
        raise RagError(ErrorCode.CONFIGURATION_INVALID)
    return dependencies


def _object_policy(request: Request) -> GcsObjectPolicy:
    settings = request.app.state.settings
    if settings is None:
        raise RagError(ErrorCode.CONFIGURATION_INVALID)
    return GcsObjectPolicy(
        bucket=settings.gcs_document_bucket,
        tenant_id=settings.dev_tenant_id,
        max_bytes=settings.max_document_bytes,
    )


def _perform_registration(
    request: Request,
    body: RegisterDocumentRequest,
) -> DocumentResponse:
    settings = request.app.state.settings
    dependencies = _dependencies(request)
    if settings is None:
        raise RagError(ErrorCode.CONFIGURATION_INVALID)
    if body.corpus != settings.fixed_corpus:
        raise RagError(ErrorCode.ACCESS_DENIED)
    object_policy = _object_policy(request)
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


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=DocumentResponse,
    operation_id="registerDocument",
)
def register_document(body: RegisterDocumentRequest, request: Request) -> DocumentResponse:
    return _perform_registration(request, body)


@router.post(
    "/upload",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=DocumentResponse,
    operation_id="uploadDocument",
)
async def upload_document(
    request: Request,
    file: Annotated[UploadFile, File()],
    document_id: Annotated[str | None, Form()] = None,
) -> DocumentResponse:
    settings = request.app.state.settings
    dependencies = _dependencies(request)
    if settings is None:
        raise RagError(ErrorCode.CONFIGURATION_INVALID)
    filename = PurePath(file.filename or "upload.txt").name
    suffix = PurePath(filename).suffix.lower()
    media_type = _MEDIA_BY_SUFFIX.get(suffix)
    if media_type is None:
        raise RagError(ErrorCode.UNSUPPORTED_MEDIA_TYPE)
    doc_id = _resolve_document_id(filename, document_id)
    data = await file.read()
    if not data:
        raise RagError(ErrorCode.INVALID_REQUEST)
    if len(data) > settings.max_document_bytes:
        raise RagError(ErrorCode.DOCUMENT_TOO_LARGE)
    checksum = hashlib.sha256(data).hexdigest()
    object_name = f"{settings.tenant_upload_prefix}/{doc_id}{suffix}"
    stored = dependencies.objects.write_new(
        object_name=object_name,
        data=data,
        content_type=media_type,
        checksum_sha256=checksum,
        deadline=settings.dependency_timeout_seconds,
    )
    body = RegisterDocumentRequest(
        document_id=doc_id,
        object_uri=stored.canonical_uri,
        object_generation=stored.object_generation,
        media_type=media_type,
        checksum_sha256=checksum,
        idempotency_key=f"upload-{doc_id}-{stored.object_generation}",
    )
    return _perform_registration(request, body)


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
