"""Deterministic, non-production embedding/vector/generation HTTP substitute."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

MAX_BODY_BYTES = 2_000_000
REQUIRED_DATAPOINT_FIELDS = frozenset(
    {
        "vector_id",
        "vector",
        "tenant_id",
        "corpus",
        "document_id",
        "generation",
        "active_generation",
        "index_version",
    }
)
REQUIRED_RESTRICTIONS = frozenset({"tenant_id", "corpus", "index_version", "active_generations"})


class ApiError(Exception):
    """A sanitized HTTP error."""

    def __init__(self, status: HTTPStatus, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail


def deterministic_embedding(text: str, dimensions: int) -> list[float]:
    """Produce deterministic local embeddings."""
    if not text or dimensions < 1 or dimensions > 10_000:
        raise ApiError(HTTPStatus.BAD_REQUEST, "INVALID_REQUEST", "invalid embedding input")
    digest = hashlib.sha256(text.encode()).digest()
    vector = [((digest[index % len(digest)] / 255.0) * 2) - 1 for index in range(dimensions)]
    magnitude = math.sqrt(sum(value * value for value in vector))
    return [value / magnitude for value in vector]


def _finite_vector(value: object) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ApiError(HTTPStatus.BAD_REQUEST, "INVALID_REQUEST", "vector must be non-empty")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item)
        for item in value
    ):
        raise ApiError(HTTPStatus.BAD_REQUEST, "INVALID_REQUEST", "vector must be finite")
    return [float(item) for item in value]


class EmulatorStore:
    """SQLite-backed deterministic vectors and one-shot failure injection."""

    def __init__(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS vectors (
                    vector_id TEXT PRIMARY KEY,
                    vector TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    corpus TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    active_generation TEXT NOT NULL,
                    index_version TEXT NOT NULL
                )
                """
            )
        self._failures: dict[str, list[tuple[int, str]]] = {}

    def inject_failure(self, operation: str, count: int, code: str) -> None:
        if operation not in {"embed", "upsert", "delete", "search", "generate"}:
            raise ApiError(HTTPStatus.BAD_REQUEST, "INVALID_REQUEST", "unknown operation")
        if count < 1 or count > 20 or not code:
            raise ApiError(HTTPStatus.BAD_REQUEST, "INVALID_REQUEST", "invalid failure")
        self._failures.setdefault(operation, []).append((count, code))

    def maybe_fail(self, operation: str) -> None:
        failures = self._failures.get(operation, [])
        if not failures:
            return
        count, code = failures[0]
        if count == 1:
            failures.pop(0)
        else:
            failures[0] = (count - 1, code)
        raise ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            code,
            "injected deterministic dependency failure",
        )

    def reset(self) -> None:
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM vectors")
            self._failures.clear()

    def upsert(self, datapoints: object) -> int:
        if not isinstance(datapoints, list) or not datapoints or len(datapoints) > 250:
            raise ApiError(HTTPStatus.BAD_REQUEST, "INVALID_REQUEST", "invalid datapoints")
        rows: list[tuple[object, ...]] = []
        dimension: int | None = None
        for item in datapoints:
            if not isinstance(item, dict) or set(item) != REQUIRED_DATAPOINT_FIELDS:
                raise ApiError(HTTPStatus.BAD_REQUEST, "INVALID_REQUEST", "invalid datapoint")
            vector = _finite_vector(item["vector"])
            dimension = dimension or len(vector)
            generation = item["generation"]
            document_id = item["document_id"]
            expected_active = f"{len(str(document_id))}:{document_id}:{generation}"
            if (
                len(vector) != dimension
                or not isinstance(generation, int)
                or generation < 1
                or item["active_generation"] != expected_active
                or any(
                    not isinstance(item[field], str) or not item[field]
                    for field in (
                        "vector_id",
                        "tenant_id",
                        "corpus",
                        "document_id",
                        "active_generation",
                        "index_version",
                    )
                )
            ):
                raise ApiError(HTTPStatus.BAD_REQUEST, "INVALID_REQUEST", "invalid datapoint")
            rows.append(
                (
                    item["vector_id"],
                    json.dumps(vector, separators=(",", ":")),
                    item["tenant_id"],
                    item["corpus"],
                    item["document_id"],
                    generation,
                    item["active_generation"],
                    item["index_version"],
                )
            )
        with self._lock, self._connection:
            self._connection.executemany(
                """
                INSERT INTO vectors VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(vector_id) DO UPDATE SET
                    vector=excluded.vector,
                    tenant_id=excluded.tenant_id,
                    corpus=excluded.corpus,
                    document_id=excluded.document_id,
                    generation=excluded.generation,
                    active_generation=excluded.active_generation,
                    index_version=excluded.index_version
                """,
                rows,
            )
        return len(rows)

    def delete(self, vector_ids: object) -> int:
        if (
            not isinstance(vector_ids, list)
            or not vector_ids
            or len(vector_ids) > 250
            or any(not isinstance(item, str) or not item for item in vector_ids)
        ):
            raise ApiError(HTTPStatus.BAD_REQUEST, "INVALID_REQUEST", "invalid vector IDs")
        with self._lock, self._connection:
            before = self._connection.total_changes
            self._connection.executemany(
                "DELETE FROM vectors WHERE vector_id = ?", ((item,) for item in vector_ids)
            )
            return self._connection.total_changes - before

    def read(self, vector_ids: object, limit: object) -> list[dict[str, object]]:
        if (
            not isinstance(vector_ids, list)
            or not vector_ids
            or not isinstance(limit, int)
            or isinstance(limit, bool)
            or limit < len(vector_ids)
            or limit > 250
            or len(set(vector_ids)) != len(vector_ids)
            or any(not isinstance(item, str) or not item for item in vector_ids)
        ):
            raise ApiError(HTTPStatus.BAD_REQUEST, "INVALID_REQUEST", "invalid vector read")
        with self._lock:
            rows = [
                row
                for vector_id in vector_ids
                if (
                    row := self._connection.execute(
                        "SELECT * FROM vectors WHERE vector_id = ?", (vector_id,)
                    ).fetchone()
                )
                is not None
            ]
        found = {
            row["vector_id"]: {
                "vector_id": row["vector_id"],
                "vector": json.loads(row["vector"]),
                "tenant_id": row["tenant_id"],
                "corpus": row["corpus"],
                "document_id": row["document_id"],
                "generation": row["generation"],
                "active_generation": row["active_generation"],
                "index_version": row["index_version"],
            }
            for row in rows
        }
        return [found[item] for item in vector_ids if item in found]

    def search(
        self, vector_value: object, restrictions: object, top_k: object
    ) -> list[dict[str, object]]:
        vector = _finite_vector(vector_value)
        if (
            not isinstance(restrictions, dict)
            or set(restrictions) != REQUIRED_RESTRICTIONS
            or any(
                not isinstance(restrictions[field], str) or not restrictions[field]
                for field in ("tenant_id", "corpus", "index_version")
            )
            or not isinstance(restrictions["active_generations"], list)
            or not restrictions["active_generations"]
            or any(
                not isinstance(item, str) or not item for item in restrictions["active_generations"]
            )
            or not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or top_k < 1
            or top_k > 50
        ):
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                "MISSING_RESTRICTION",
                "all exact-match restrictions are mandatory",
            )
        query = """
            SELECT * FROM vectors
            WHERE tenant_id = ? AND corpus = ? AND index_version = ?
        """
        parameters = [
            restrictions["tenant_id"],
            restrictions["corpus"],
            restrictions["index_version"],
        ]
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        allowed_generations = set(restrictions["active_generations"])
        candidates = []
        for row in rows:
            if row["active_generation"] not in allowed_generations:
                continue
            stored = json.loads(row["vector"])
            if len(stored) != len(vector):
                continue
            candidates.append(
                {
                    "vector_id": row["vector_id"],
                    "score": sum(left * right for left, right in zip(vector, stored, strict=True)),
                    "document_id": row["document_id"],
                    "generation": row["generation"],
                    "chunk_id": row["vector_id"],
                }
            )
        candidates.sort(key=lambda item: (-float(item["score"]), str(item["vector_id"])))
        return candidates[:top_k]


class Handler(BaseHTTPRequestHandler):
    """Small JSON HTTP facade; every response labels the emulator."""

    server: EmulatorServer

    def log_message(self, format_: str, *args: object) -> None:
        print(f"local_vertex_emulator {self.command} {self.path} {args[1]}", flush=True)

    def _json(self, status: HTTPStatus, body: dict[str, object]) -> None:
        payload = json.dumps(
            {"non_production": True, **body}, separators=(",", ":"), sort_keys=True
        ).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("X-RAG-Emulator", "non-production")
        self.end_headers()
        self.wfile.write(payload)

    def _body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ApiError(HTTPStatus.BAD_REQUEST, "INVALID_REQUEST", "invalid body") from error
        if length < 1 or length > MAX_BODY_BYTES:
            raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "LIMIT_EXCEEDED", "invalid body")
        try:
            result = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ApiError(HTTPStatus.BAD_REQUEST, "INVALID_REQUEST", "invalid JSON") from error
        if not isinstance(result, dict):
            raise ApiError(HTTPStatus.BAD_REQUEST, "INVALID_REQUEST", "object required")
        return result

    def _admin(self) -> None:
        supplied = self.headers.get("X-Local-Admin-Token", "")
        if supplied != self.server.admin_token:
            raise ApiError(HTTPStatus.UNAUTHORIZED, "UNAUTHENTICATED", "local admin token required")

    def do_GET(self) -> None:
        if self.path in {"/healthz", "/readyz"}:
            self._json(HTTPStatus.OK, {"status": "ok"})
            return
        self._json(HTTPStatus.NOT_FOUND, {"code": "NOT_FOUND"})

    def do_POST(self) -> None:
        try:
            body = self._body()
            if self.path == "/v1/embeddings":
                self.server.store.maybe_fail("embed")
                texts = body.get("texts")
                dimensions = body.get("dimensions")
                if (
                    set(body) != {"texts", "dimensions"}
                    or not isinstance(texts, list)
                    or not texts
                    or len(texts) > 250
                    or any(not isinstance(text, str) or not text for text in texts)
                    or not isinstance(dimensions, int)
                    or isinstance(dimensions, bool)
                ):
                    raise ApiError(
                        HTTPStatus.BAD_REQUEST, "INVALID_REQUEST", "invalid embedding request"
                    )
                self._json(
                    HTTPStatus.OK,
                    {"vectors": [deterministic_embedding(text, dimensions) for text in texts]},
                )
            elif self.path == "/v1/vectors:upsert":
                self.server.store.maybe_fail("upsert")
                if set(body) != {"datapoints"}:
                    raise ApiError(HTTPStatus.BAD_REQUEST, "INVALID_REQUEST", "invalid upsert")
                self._json(
                    HTTPStatus.OK, {"upserted": self.server.store.upsert(body["datapoints"])}
                )
            elif self.path == "/v1/vectors:delete":
                self.server.store.maybe_fail("delete")
                if set(body) != {"vector_ids"}:
                    raise ApiError(HTTPStatus.BAD_REQUEST, "INVALID_REQUEST", "invalid delete")
                self._json(HTTPStatus.OK, {"deleted": self.server.store.delete(body["vector_ids"])})
            elif self.path == "/v1/vectors:search":
                self.server.store.maybe_fail("search")
                if set(body) != {"vector", "restrictions", "top_k"}:
                    raise ApiError(HTTPStatus.BAD_REQUEST, "INVALID_REQUEST", "invalid search")
                self._json(
                    HTTPStatus.OK,
                    {
                        "candidates": self.server.store.search(
                            body["vector"], body["restrictions"], body["top_k"]
                        )
                    },
                )
            elif self.path == "/v1/vectors:read":
                self.server.store.maybe_fail("search")
                if set(body) != {"vector_ids", "limit"}:
                    raise ApiError(HTTPStatus.BAD_REQUEST, "INVALID_REQUEST", "invalid read")
                self._json(
                    HTTPStatus.OK,
                    {"datapoints": self.server.store.read(body["vector_ids"], body["limit"])},
                )
            elif self.path == "/v1/generation":
                self.server.store.maybe_fail("generate")
                question = body.get("question")
                context = body.get("authorized_context")
                max_tokens = body.get("max_output_tokens")
                if (
                    set(body) != {"question", "authorized_context", "max_output_tokens"}
                    or not isinstance(question, str)
                    or not question
                    or not isinstance(context, list)
                    or not context
                    or not isinstance(max_tokens, int)
                    or isinstance(max_tokens, bool)
                    or max_tokens < 1
                    or any(
                        not isinstance(item, dict)
                        or set(item) != {"citation_id", "text"}
                        or not isinstance(item["citation_id"], str)
                        or not item["citation_id"]
                        or not isinstance(item["text"], str)
                        or not item["text"]
                        for item in context
                    )
                ):
                    raise ApiError(
                        HTTPStatus.BAD_REQUEST, "INVALID_REQUEST", "authorized evidence required"
                    )
                first = context[0]
                citations = (
                    []
                    if "lunar launch code" in question.casefold()
                    else [
                        {
                            "citation_id": first["citation_id"],
                            "quote": first["text"][:2000],
                        }
                    ]
                )
                self._json(
                    HTTPStatus.OK,
                    {
                        "answer": "Grounded answer." if citations else "No grounded answer.",
                        "citations": citations,
                    },
                )
            elif self.path == "/_test/failures":
                self._admin()
                if set(body) != {"operation", "count", "code"}:
                    raise ApiError(HTTPStatus.BAD_REQUEST, "INVALID_REQUEST", "invalid failure")
                self.server.store.inject_failure(body["operation"], body["count"], body["code"])
                self._json(HTTPStatus.OK, {"configured": True})
            elif self.path == "/_test/reset":
                self._admin()
                self.server.store.reset()
                self._json(HTTPStatus.OK, {"reset": True})
            else:
                self._json(HTTPStatus.NOT_FOUND, {"code": "NOT_FOUND"})
        except ApiError as error:
            self._json(error.status, {"code": error.code, "detail": error.detail})


class EmulatorServer(ThreadingHTTPServer):
    """Typed server state exposed to request handlers."""

    def __init__(self, address: tuple[str, int], store: EmulatorStore, admin_token: str) -> None:
        super().__init__(address, Handler)
        self.store = store
        self.admin_token = admin_token


def main() -> None:
    """Run only under an explicitly local, non-production profile."""
    if os.environ.get("APP_MODE") != "local":
        raise SystemExit("local Vertex/vector emulator requires APP_MODE=local")
    admin_token = os.environ.get("LOCAL_EMULATOR_ADMIN_TOKEN", "")
    if len(admin_token) < 16:
        raise SystemExit("LOCAL_EMULATOR_ADMIN_TOKEN must be explicit and at least 16 characters")
    path = os.environ.get("LOCAL_EMULATOR_DB", "/data/local-emulator.sqlite3")
    port = int(os.environ.get("PORT", "8090"))
    server = EmulatorServer(
        ("0.0.0.0", port),  # nosec B104  # noqa: S104 - container namespace only
        EmulatorStore(path),
        admin_token,
    )
    print(f"non-production local Vertex/vector emulator listening on {port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
