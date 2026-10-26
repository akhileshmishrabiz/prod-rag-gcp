"""Worker-owned dependency-free liveness and lightweight readiness probes."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from packages.rag_core.errors import ConfigurationError, ErrorCode, RagError

router = APIRouter(tags=["probes"])


def _problem(trace_id: str = "readiness") -> dict[str, object]:
    return {
        "type": "https://example.invalid/problems/configuration-invalid",
        "title": "Service configuration is invalid",
        "status": 503,
        "code": "CONFIGURATION_INVALID",
        "detail": "Readiness checks failed; inspect sanitized service logs.",
        "trace_id": trace_id,
        "request_id": trace_id,
        "retryable": False,
    }


@router.get("/healthz")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz", response_model=None)
async def readiness(request: Request) -> dict[str, str] | JSONResponse:
    error: ConfigurationError | None = request.app.state.configuration_error
    if error is not None:
        return JSONResponse(
            status_code=503,
            content=error.as_problem(),
            media_type="application/problem+json",
        )
    settings = request.app.state.settings
    dependencies = request.app.state.dependencies
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
            return JSONResponse(
                status_code=503,
                content=_problem(),
                media_type="application/problem+json",
            )
    return {"status": "ok"}
