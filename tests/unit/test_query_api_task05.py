"""Offline acceptance coverage for the API and retrieval policy."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
import yaml  # type: ignore[import-untyped]
from fastapi.testclient import TestClient

from packages.rag_core.config import AppMode, AuthMode
from packages.rag_core.errors import ErrorCode, RagError
from packages.rag_core.models import (
    ActiveIndexRecord,
    ChunkRecord,
    DocumentRecord,
    DocumentStatus,
    GeneratedAnswer,
    GeneratedCitation,
    MediaType,
    ObjectMetadata,
    VectorRestrictions,
    chunk_ids_sha256,
)
from packages.rag_core.policy import build_datapoint
from packages.rag_core.retrieval import RetrievalLimits, RetrievalService
from services.api.app.auth import issue_local_token
from services.api.app.dependencies import ApiDependencies, ApiPolicy
from services.api.app.main import create_app
from services.api.app.openapi import semantic_surface
from tests.fakes import (
    FakeClock,
    FakeEmbeddingModel,
    FakeEventPublisher,
    FakeGenerator,
    FakeIds,
    FakeMetadataStore,
    FakeObjectStore,
    FakeVectorIndex,
)


def _dependencies() -> tuple[ApiDependencies, FakeMetadataStore, FakeVectorIndex]:
    now = datetime(2025, 1, 1, tzinfo=UTC)
    index = ActiveIndexRecord(
        index_version="index-1",
        pipeline_version="pipeline-1",
        parser_version="parser-1",
        chunker_version="chunker-1",
        embedding_model_id="deterministic-embedding-v1",
        embedding_dimensions=32,
        updated_at=now,
    )
    metadata = FakeMetadataStore(tenant_id="dev-tenant", active_index=index)
    vectors = FakeVectorIndex()
    embeddings = FakeEmbeddingModel()
    retrieval = RetrievalService(
        metadata=metadata,
        embeddings=embeddings,
        vectors=vectors,
        generator=FakeGenerator(),
        tenant_id="dev-tenant",
        pipeline_version="pipeline-1",
        parser_version="parser-1",
        chunker_version="chunker-1",
        generation_model_id="deterministic-generator-v1",
        embedding_model_id="deterministic-embedding-v1",
        embedding_dimensions=32,
        limits=RetrievalLimits(
            max_active_generations=100,
            max_candidates_per_document=2,
            max_context_chars=1000,
            max_context_tokens=250,
            max_evidence_chars=1000,
            max_output_tokens=128,
            dependency_timeout_seconds=5,
        ),
    )
    dependencies = ApiDependencies(
        metadata=metadata,
        objects=FakeObjectStore(),
        publisher=FakeEventPublisher(),
        retrieval=retrieval,
        clock=FakeClock(now),
        ids=FakeIds(),
        policy=ApiPolicy(),
    )
    return dependencies, metadata, vectors


def _client(
    local_settings: Any, tmp_path: Path, dependencies: ApiDependencies
) -> tuple[TestClient, dict[str, str]]:
    key_path = tmp_path / "local-auth.key"
    key = b"offline-test-signing-key"
    key_path.write_bytes(key)
    settings = local_settings.model_copy(update={"local_auth_signing_key_file": key_path})
    token = issue_local_token(
        key=key,
        identity=dependencies.policy.local_test_identity,
        tenant_id=settings.dev_tenant_id,
        expires_at=dependencies.clock.now() + timedelta(minutes=5),
    )
    return (
        TestClient(create_app(lambda: settings, dependencies)),
        {"Authorization": f"Bearer {token}"},
    )


def _add_ready_evidence(
    metadata: FakeMetadataStore, vectors: FakeVectorIndex, question: str
) -> None:
    now = datetime(2025, 1, 1, tzinfo=UTC)
    text = "The service answers only from authorized evidence."
    vector_id = "vector-1"
    metadata.documents["doc-1"] = DocumentRecord(
        tenant_id="dev-tenant",
        document_id="doc-1",
        corpus="default",
        object_uri="gs://dev-rag-documents/tenants/dev-tenant/uploads/doc-1.txt",
        object_generation=7,
        checksum_sha256="a" * 64,
        media_type=MediaType.TEXT,
        pipeline_version="pipeline-1",
        generation=1,
        status=DocumentStatus.READY,
        tombstoned=False,
        created_at=now,
        updated_at=now,
        chunk_count=1,
        expected_chunk_ids_sha256=chunk_ids_sha256([vector_id]),
        parser_version="parser-1",
        chunker_version="chunker-1",
        embedding_model_id="deterministic-embedding-v1",
        embedding_dimensions=32,
        active_index_version="index-1",
        ready_at=now,
    )
    metadata.chunks[("doc-1", 1, vector_id)] = ChunkRecord(
        tenant_id="dev-tenant",
        corpus="default",
        document_id="doc-1",
        generation=1,
        object_generation=7,
        chunk_id=vector_id,
        ordinal=0,
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        vector_id=vector_id,
        pipeline_version="pipeline-1",
        parser_version="parser-1",
        chunker_version="chunker-1",
        staged_at=now,
        created_at=now,
    )
    vector = FakeEmbeddingModel().embed([question], 32, 5)[0]
    vectors.upsert(
        [
            build_datapoint(
                vector_id=vector_id,
                vector=vector,
                tenant_id="dev-tenant",
                corpus="default",
                document_id="doc-1",
                generation=1,
                index_version="index-1",
            )
        ],
        5,
    )


def test_local_auth_is_signed_and_tenant_is_not_caller_selectable(
    local_settings: Any, tmp_path: Path
) -> None:
    dependencies, _, _ = _dependencies()
    client, headers = _client(local_settings, tmp_path, dependencies)
    assert client.post("/v1/query", json={"question": "hello"}).status_code == 401
    response = client.post(
        "/v1/query",
        headers=headers,
        json={"tenant_id": "other", "question": "hello"},
    )
    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_REQUEST"


def test_query_uses_restrictions_and_returns_verified_citation(
    local_settings: Any, tmp_path: Path
) -> None:
    dependencies, metadata, vectors = _dependencies()
    question = "What is the policy?"
    _add_ready_evidence(metadata, vectors, question)
    client, headers = _client(local_settings, tmp_path, dependencies)
    response = client.post(
        "/v1/query",
        headers=headers,
        json={"question": question, "top_k": 4},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["abstained"] is False
    assert body["model_version"] == "deterministic-generator-v1"
    assert body["index_generation"] == "index-1"
    assert body["citations"][0]["source_uri"].startswith("gs://dev-rag-documents/")
    assert "page" not in body["citations"][0]
    assert body["citations"][0]["document_id"] == "doc-1"
    search = next(arguments for method, arguments in vectors.calls if method == "search")
    restrictions = cast(VectorRestrictions, search["restrictions"])
    assert restrictions.tenant_id == "dev-tenant"
    assert restrictions.corpus == "default"
    assert restrictions.index_version == "index-1"
    assert restrictions.active_generations


def test_no_active_generation_abstains_before_vector_call(
    local_settings: Any, tmp_path: Path
) -> None:
    dependencies, _, vectors = _dependencies()
    client, headers = _client(local_settings, tmp_path, dependencies)
    response = client.post(
        "/v1/query",
        headers=headers,
        json={"question": "unknown"},
    )
    assert response.status_code == 200
    assert response.json()["abstained"] is True
    assert response.json()["answer"] == ""
    assert not any(method == "search" for method, _ in vectors.calls)


def test_cross_tenant_hydrated_chunk_is_discarded(local_settings: Any, tmp_path: Path) -> None:
    dependencies, metadata, vectors = _dependencies()
    question = "What is the policy?"
    _add_ready_evidence(metadata, vectors, question)
    original = metadata.chunks[("doc-1", 1, "vector-1")]
    metadata.chunks[("doc-1", 1, "vector-1")] = original.model_copy(
        update={"tenant_id": "other-tenant"}
    )
    client, headers = _client(local_settings, tmp_path, dependencies)
    response = client.post(
        "/v1/query",
        headers=headers,
        json={"question": question},
    )
    assert response.status_code == 200
    assert response.json()["abstained"] is True


def test_tombstone_blocks_query_before_delete_processing(
    local_settings: Any, tmp_path: Path
) -> None:
    dependencies, metadata, vectors = _dependencies()
    question = "What is the policy?"
    _add_ready_evidence(metadata, vectors, question)
    metadata.tombstone("dev-tenant", "doc-1", dependencies.clock.now())
    vectors.calls.clear()
    client, headers = _client(local_settings, tmp_path, dependencies)
    response = client.post(
        "/v1/query",
        headers=headers,
        json={"question": question},
    )
    assert response.status_code == 200
    assert response.json()["abstained"] is True
    assert not any(method == "search" for method, _ in vectors.calls)


class _UnsupportedCitationGenerator(FakeGenerator):
    def generate(self, *args: Any, **kwargs: Any) -> GeneratedAnswer:
        return GeneratedAnswer(
            answer="unsupported",
            citations=[GeneratedCitation(citation_id="not-authorized", quote="invented")],
        )


def test_unsupported_citation_forces_abstention(local_settings: Any, tmp_path: Path) -> None:
    dependencies, metadata, vectors = _dependencies()
    question = "What is the policy?"
    _add_ready_evidence(metadata, vectors, question)
    retrieval = dependencies.retrieval
    retrieval.generator = _UnsupportedCitationGenerator()
    client, headers = _client(local_settings, tmp_path, dependencies)
    response = client.post(
        "/v1/query",
        headers=headers,
        json={"question": question},
    )
    assert response.status_code == 200
    assert response.json()["abstained"] is True
    assert response.json()["citations"] == []


def test_document_registration_get_and_tombstone(local_settings: Any, tmp_path: Path) -> None:
    dependencies, metadata, _ = _dependencies()
    content = b"registered content"
    checksum = hashlib.sha256(content).hexdigest()
    uri = "gs://dev-rag-documents/tenants/dev-tenant/uploads/doc.txt"
    assert isinstance(dependencies.objects, FakeObjectStore)
    dependencies.objects.put(
        ObjectMetadata(
            canonical_uri=uri,
            bucket="dev-rag-documents",
            name="tenants/dev-tenant/uploads/doc.txt",
            object_generation=7,
            size=len(content),
            content_type=MediaType.TEXT,
            checksum_sha256=checksum,
            updated_at=datetime(2025, 1, 1, tzinfo=UTC),
        ),
        content,
    )
    client, headers = _client(local_settings, tmp_path, dependencies)
    response = client.post(
        "/v1/documents",
        headers=headers,
        json={
            "document_id": "doc-api",
            "object_uri": uri,
            "object_generation": 7,
            "media_type": "text/plain",
            "checksum_sha256": checksum,
            "idempotency_key": "registration-1",
        },
    )
    assert response.status_code == 202
    assert client.get("/v1/documents/doc-api", headers=headers).status_code == 200
    deleted = client.delete("/v1/documents/doc-api", headers=headers)
    assert deleted.status_code == 202
    assert deleted.json()["status"] == "DELETE_PENDING"
    assert metadata.documents["doc-api"].tombstoned is True
    assert metadata.list_active_generations("dev-tenant", "default", 100) == []


def test_generated_openapi_semantically_matches_contract(local_settings: Any) -> None:
    app = create_app(lambda: local_settings)
    schema = app.openapi()
    contract_path = Path(__file__).parents[2] / "contracts" / "openapi" / "openapi.v1.yaml"
    contract = cast(dict[str, Any], yaml.safe_load(contract_path.read_text()))
    assert semantic_surface(schema) == semantic_surface(contract)
    assert "tenant_id" not in str(schema).lower()


@pytest.mark.parametrize(
    "update",
    [
        {"max_active_generations": 0},
        {"max_candidates_per_document": 0},
        {"max_context_chars": 0},
        {"max_context_tokens": 0},
        {"max_evidence_chars": 0},
        {"max_output_tokens": 0},
        {"dependency_timeout_seconds": 0},
    ],
)
def test_retrieval_limits_reject_nonpositive_values(update: dict[str, int]) -> None:
    values: dict[str, int | float] = {
        "max_active_generations": 1,
        "max_candidates_per_document": 1,
        "max_context_chars": 1,
        "max_context_tokens": 1,
        "max_evidence_chars": 1,
        "max_output_tokens": 1,
        "dependency_timeout_seconds": 1,
    }
    values.update(update)
    with pytest.raises(ValueError, match="retrieval limits must be positive"):
        RetrievalLimits(**values)  # type: ignore[arg-type]


def test_retrieval_rejects_invalid_query_and_incompatible_index() -> None:
    dependencies, metadata, _ = _dependencies()
    with pytest.raises(RagError) as invalid:
        dependencies.retrieval.query(question=" ", corpus="default", top_k=0, trace_id="trace")
    assert invalid.value.code is ErrorCode.INVALID_REQUEST
    metadata.active_index = metadata.active_index.model_copy(
        update={"parser_version": "other-parser"}
    )
    with pytest.raises(RagError) as incompatible:
        dependencies.retrieval.query(
            question="question", corpus="default", top_k=1, trace_id="trace"
        )
    assert incompatible.value.code is ErrorCode.CONFIGURATION_INVALID


def test_empty_vector_results_and_stale_parent_abstain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependencies, metadata, vectors = _dependencies()
    question = "What is the policy?"
    _add_ready_evidence(metadata, vectors, question)
    vectors.datapoints.clear()
    result = dependencies.retrieval.query(
        question=question, corpus="default", top_k=2, trace_id="trace"
    )
    assert result.abstained is True
    _add_ready_evidence(metadata, vectors, question)
    monkeypatch.setattr(metadata, "get_document", lambda *_args: None)
    result = dependencies.retrieval.query(
        question=question, corpus="default", top_k=2, trace_id="trace"
    )
    assert result.abstained is True


class _NoCitationGenerator(FakeGenerator):
    def generate(self, *args: Any, **kwargs: Any) -> GeneratedAnswer:
        return GeneratedAnswer(answer="uncited", citations=[])


def test_empty_citations_abstain_and_context_bounds_apply() -> None:
    dependencies, metadata, vectors = _dependencies()
    question = "What is the policy?"
    _add_ready_evidence(metadata, vectors, question)
    dependencies.retrieval.generator = _NoCitationGenerator()
    result = dependencies.retrieval.query(
        question=question, corpus="default", top_k=2, trace_id="trace"
    )
    assert result.abstained is True

    original = metadata.chunks[("doc-1", 1, "vector-1")]
    adjacent = original.model_copy(
        update={"chunk_id": "adjacent", "vector_id": "adjacent", "ordinal": 1}
    )
    separated = original.model_copy(
        update={"chunk_id": "separated", "vector_id": "separated", "ordinal": 2}
    )
    over_cap = original.model_copy(
        update={"chunk_id": "over-cap", "vector_id": "over-cap", "ordinal": 4}
    )
    oversized = original.model_copy(update={"text": "x" * 2000, "ordinal": 8})
    bounded = dependencies.retrieval._bound_evidence(
        [original, adjacent, separated, over_cap, oversized]
    )
    assert [chunk.ordinal for chunk in bounded] == [0, 2]


def test_api_validation_auth_and_dependency_error_paths(
    local_settings: Any, tmp_path: Path
) -> None:
    dependencies, _, _ = _dependencies()
    client, headers = _client(local_settings, tmp_path, dependencies)
    assert (
        client.post(
            "/v1/query",
            headers={"Authorization": "Bearer malformed"},
            json={"question": "hello"},
        ).status_code
        == 401
    )
    assert client.post("/v1/query", headers=headers, json={"question": "   "}).status_code == 400
    assert (
        client.post(
            "/v1/query",
            headers=headers,
            json={"question": "x" * (local_settings.max_query_input_chars + 1)},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/v1/query", headers=headers, json={"question": "hello", "top_k": 21}
        ).status_code
        == 422
    )
    assert client.get("/v1/documents/missing", headers=headers).status_code == 404

    cloud_settings = local_settings.model_copy(
        update={
            "app_mode": AppMode.GCP,
            "auth_mode": AuthMode.CLOUD_RUN_IAM,
            "local_auth_signing_key_file": None,
        }
    )
    cloud_client = TestClient(create_app(lambda: cloud_settings, dependencies))
    assert cloud_client.post("/v1/query", json={"question": "hello"}).status_code == 200


def test_registration_rejects_prefix_generation_and_media_mismatches(
    local_settings: Any, tmp_path: Path
) -> None:
    dependencies, _, _ = _dependencies()
    content = b"registered content"
    checksum = hashlib.sha256(content).hexdigest()
    assert isinstance(dependencies.objects, FakeObjectStore)
    wrong_prefix = "gs://dev-rag-documents/tenants/dev-tenant/not-uploads/doc.txt"
    valid_uri = "gs://dev-rag-documents/tenants/dev-tenant/uploads/doc.txt"
    for uri, name in (
        (wrong_prefix, "tenants/dev-tenant/not-uploads/doc.txt"),
        (valid_uri, "tenants/dev-tenant/uploads/doc.txt"),
    ):
        dependencies.objects.put(
            ObjectMetadata(
                canonical_uri=uri,
                bucket="dev-rag-documents",
                name=name,
                object_generation=7,
                size=len(content),
                content_type=MediaType.TEXT,
                checksum_sha256=checksum,
                updated_at=dependencies.clock.now(),
            ),
            content,
        )
    client, headers = _client(local_settings, tmp_path, dependencies)
    base = {
        "document_id": "doc-api",
        "object_generation": 7,
        "media_type": "text/plain",
        "checksum_sha256": checksum,
        "idempotency_key": "registration-1",
    }
    assert (
        client.post(
            "/v1/documents",
            headers=headers,
            json={**base, "object_uri": wrong_prefix},
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/v1/documents",
            headers=headers,
            json={**base, "object_uri": valid_uri, "object_generation": 8},
        ).status_code
        == 409
    )
    assert (
        client.post(
            "/v1/documents",
            headers=headers,
            json={**base, "object_uri": valid_uri, "media_type": "application/pdf"},
        ).status_code
        == 415
    )


def test_readiness_rejects_incompatible_active_index(
    local_settings: Any,
) -> None:
    dependencies, metadata, _ = _dependencies()
    metadata.active_index = metadata.active_index.model_copy(update={"chunker_version": "other"})
    response = TestClient(create_app(lambda: local_settings, dependencies)).get("/readyz")
    assert response.status_code == 503
    assert response.json()["code"] == "CONFIGURATION_INVALID"
