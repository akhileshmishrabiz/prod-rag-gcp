"""Sanitized problem-details boundary and request correlation."""

from __future__ import annotations

import re
import secrets
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from packages.rag_core.errors import ErrorCode, RagError

_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:/-]{1,128}$")
_STATUS = {
    ErrorCode.INVALID_REQUEST: 400,
    ErrorCode.AUTHENTICATION_REQUIRED: 401,
    ErrorCode.ACCESS_DENIED: 403,
    ErrorCode.DOCUMENT_NOT_FOUND: 404,
    ErrorCode.DOCUMENT_CONFLICT: 409,
    ErrorCode.DOCUMENT_TOO_LARGE: 413,
    ErrorCode.UNSUPPORTED_MEDIA_TYPE: 415,
    ErrorCode.UNPROCESSABLE_DOCUMENT: 422,
    ErrorCode.LIMIT_EXCEEDED: 422,
    ErrorCode.RATE_LIMITED: 429,
    ErrorCode.STALE_GENERATION: 409,
    ErrorCode.DUPLICATE_EVENT: 200,
    ErrorCode.LEASE_CONTENDED: 409,
    ErrorCode.CONTRACT_VERSION_UNSUPPORTED: 422,
    ErrorCode.TENANT_MISMATCH: 403,
    ErrorCode.DEPENDENCY_TIMEOUT: 503,
    ErrorCode.DEPENDENCY_UNAVAILABLE: 503,
    ErrorCode.DEPENDENCY_REJECTED: 502,
    ErrorCode.CONFIGURATION_INVALID: 503,
    ErrorCode.INTERNAL: 500,
}


def install_error_boundary(app: FastAPI) -> None:
    @app.middleware("http")
    async def correlation(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        supplied = request.headers.get("x-request-id", "")
        request.state.request_id = supplied if _REQUEST_ID.fullmatch(supplied) else _new_id()
        dependencies = getattr(request.app.state, "dependencies", None)
        request.state.trace_id = (
            dependencies.ids.new_trace_id() if dependencies is not None else _new_id()
        )
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(RagError)
    async def rag_error(request: Request, error: RagError) -> JSONResponse:
        return _problem(request, error.code, error.retryable)

    @app.exception_handler(RequestValidationError)
    async def request_error(request: Request, _error: RequestValidationError) -> JSONResponse:
        return _problem(request, ErrorCode.INVALID_REQUEST, False)

    @app.exception_handler(Exception)
    async def unknown_error(request: Request, _error: Exception) -> JSONResponse:
        return _problem(request, ErrorCode.INTERNAL, False)


def _problem(request: Request, code: ErrorCode, retryable: bool) -> JSONResponse:
    status = _STATUS.get(code, 500)
    title = code.value.replace("_", " ").title()
    return JSONResponse(
        status_code=status,
        media_type="application/problem+json",
        content={
            "type": f"https://example.invalid/problems/{code.value.lower().replace('_', '-')}",
            "title": title,
            "status": status,
            "code": code.value,
            "detail": title,
            "trace_id": getattr(request.state, "trace_id", _new_id()),
            "request_id": getattr(request.state, "request_id", _new_id()),
            "retryable": retryable,
        },
    )


def _new_id() -> str:
    return secrets.token_hex(16)
