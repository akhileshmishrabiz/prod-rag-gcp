"""Bounded fixed-tenant query endpoint."""

from fastapi import APIRouter, Depends, Request

from packages.rag_core.errors import ErrorCode, RagError
from services.api.app.auth import require_identity
from services.api.app.dependencies import ApiDependencies
from services.api.app.schemas import QueryRequest, QueryResponse

router = APIRouter(
    prefix="/v1",
    tags=["query"],
    dependencies=[Depends(require_identity)],
)


@router.post(
    "/query",
    response_model=QueryResponse,
    response_model_exclude_none=True,
    operation_id="query",
)
def query(body: QueryRequest, request: Request) -> QueryResponse:
    settings = request.app.state.settings
    dependencies: ApiDependencies | None = request.app.state.dependencies
    if settings is None or dependencies is None:
        raise RagError(ErrorCode.CONFIGURATION_INVALID)
    if body.corpus != settings.fixed_corpus:
        raise RagError(ErrorCode.ACCESS_DENIED)
    question = " ".join(body.question.split())
    if not question:
        raise RagError(ErrorCode.INVALID_REQUEST)
    if len(question) > settings.max_query_input_chars:
        raise RagError(ErrorCode.LIMIT_EXCEEDED)
    if body.top_k > settings.max_query_top_k:
        raise RagError(ErrorCode.LIMIT_EXCEEDED)
    return QueryResponse.from_result(
        dependencies.retrieval.query(
            question=question,
            corpus=body.corpus,
            top_k=body.top_k,
            trace_id=request.state.trace_id,
        )
    )
