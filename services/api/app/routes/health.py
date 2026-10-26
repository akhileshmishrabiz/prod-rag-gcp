"""Dependency-free liveness and configuration readiness probes."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from packages.rag_core.errors import ConfigurationError, ErrorCode, RagError
from services.api.app.schemas import ProbeResponse

router = APIRouter(tags=["probes"])


@router.get("/healthz", operation_id="health", response_model=ProbeResponse)
async def health() -> dict[str, str]:
    """Report process liveness without consulting configuration or dependencies."""
    return {"status": "ok"}


@router.get("/readyz", operation_id="readiness", response_model=ProbeResponse)
async def readiness(request: Request) -> dict[str, str] | JSONResponse:
    """Report configuration and lightweight active-index compatibility."""
    error: ConfigurationError | None = getattr(request.app.state, "configuration_error", None)
    if error is not None:
        return JSONResponse(
            status_code=503,
            content=error.as_problem(),
            media_type="application/problem+json",
        )
    settings = request.app.state.settings
    dependencies = getattr(request.app.state, "dependencies", None)
    if dependencies is not None:
        try:
            index = dependencies.metadata.get_active_index()
            if (
                index.pipeline_version != settings.pipeline_version
                or index.parser_version != settings.parser_version
                or index.chunker_version != settings.chunker_version
                or index.embedding_model_id != settings.embedding_model_id
                or index.embedding_dimensions != settings.embedding_dimensions
            ):
                raise RagError(ErrorCode.CONFIGURATION_INVALID)
        except RagError:
            problem = {
                "type": "https://example.invalid/problems/configuration-invalid",
                "title": "Service configuration is invalid",
                "status": 503,
                "code": "CONFIGURATION_INVALID",
                "detail": "Readiness checks failed; inspect sanitized service logs.",
                "trace_id": getattr(request.state, "trace_id", "readiness"),
                "request_id": getattr(request.state, "request_id", "readiness"),
                "retryable": False,
            }
            return JSONResponse(
                status_code=503,
                content=problem,
                media_type="application/problem+json",
            )
    return {"status": "ok"}
