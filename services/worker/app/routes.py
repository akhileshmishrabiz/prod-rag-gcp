"""Private Pub/Sub push HTTP boundary."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from packages.rag_core.errors import ErrorCode, RagError
from packages.rag_core.telemetry import resource_hash, safe_fields
from services.worker.app.auth import require_push_identity
from services.worker.app.envelope import MAX_PUSH_BODY_BYTES, decode_push_envelope
from services.worker.app.service import IngestionService

router = APIRouter(tags=["ingestion"])

_STATUS = {
    ErrorCode.AUTHENTICATION_REQUIRED: 401,
    ErrorCode.ACCESS_DENIED: 403,
    ErrorCode.LEASE_CONTENDED: 409,
    ErrorCode.RATE_LIMITED: 429,
    ErrorCode.DEPENDENCY_TIMEOUT: 503,
    ErrorCode.DEPENDENCY_UNAVAILABLE: 503,
    ErrorCode.CONFIGURATION_INVALID: 503,
}


async def _read_bounded(request: Request) -> bytes:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_PUSH_BODY_BYTES:
                raise RagError(ErrorCode.INVALID_REQUEST)
        except ValueError:
            raise RagError(ErrorCode.INVALID_REQUEST) from None
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_PUSH_BODY_BYTES:
            raise RagError(ErrorCode.INVALID_REQUEST)
    return bytes(body)


def _error_response(error: RagError, trace_id: str, request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=_STATUS.get(error.code, 500),
        content={
            "code": error.code.value,
            "retryable": error.retryable,
            "trace_id": trace_id,
            "request_id": request_id,
        },
        media_type="application/problem+json",
    )


@router.post("/v1/pubsub/push", status_code=204, response_model=None)
async def pubsub_push(request: Request) -> Response:
    trace_id = "unavailable"
    request_id = "unavailable"
    payload = None
    try:
        require_push_identity(request)
        if request.headers.get("content-type", "").partition(";")[0].strip() != "application/json":
            raise RagError(ErrorCode.INVALID_REQUEST)
        envelope, payload = decode_push_envelope(await _read_bounded(request))
        trace_id = payload.trace_id
        request_id = envelope.message.message_id
        result = IngestionService(
            settings=request.app.state.settings,
            dependencies=request.app.state.dependencies,
        ).process(payload, attempt=envelope.delivery_attempt)
        request.app.state.dependencies.telemetry.emit(
            "worker_push_ack",
            safe_fields(
                trace_id=trace_id,
                request_id=request_id,
                resource_hash=resource_hash(payload.document_id, salt=b"worker-v1"),
                operation=payload.operation.value,
                status=result.disposition,
                count=result.count,
                attempt=envelope.delivery_attempt,
                pipeline_version=payload.pipeline_version,
            ),
        )
        return Response(status_code=204)
    except RagError as error:
        dependencies = getattr(request.app.state, "dependencies", None)
        if dependencies is not None:
            dependencies.telemetry.emit(
                "worker_push_failure",
                safe_fields(
                    trace_id=trace_id,
                    request_id=request_id,
                    resource_hash=(
                        resource_hash(payload.document_id, salt=b"worker-v1")
                        if payload is not None
                        else None
                    ),
                    operation=payload.operation.value if payload is not None else None,
                    status="RETRY" if error.retryable else "ACK_TERMINAL",
                    error_code=error.code.value,
                    retryable=error.retryable,
                ),
            )
        if error.code is ErrorCode.AUTHENTICATION_REQUIRED:
            return _error_response(error, trace_id, request_id)
        if error.retryable or error.code is ErrorCode.CONFIGURATION_INVALID:
            return _error_response(error, trace_id, request_id)
        # Poison messages are acknowledged without reflecting content or details.
        return Response(status_code=204)
