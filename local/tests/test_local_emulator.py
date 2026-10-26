from __future__ import annotations

import math

import pytest

from local.emulator.app import ApiError, EmulatorStore, deterministic_embedding


def datapoint(vector_id: str = "vector-1") -> dict[str, object]:
    return {
        "vector_id": vector_id,
        "vector": [1.0, 0.0],
        "tenant_id": "dev-tenant",
        "corpus": "engineering",
        "document_id": "handbook",
        "generation": 1,
        "active_generation": "8:handbook:1",
        "index_version": "index-1",
    }


def restrictions(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "tenant_id": "dev-tenant",
        "corpus": "engineering",
        "index_version": "index-1",
        "active_generations": ["8:handbook:1"],
    }
    value.update(changes)
    return value


def test_embedding_is_normalized_and_deterministic() -> None:
    first = deterministic_embedding("fixed text", 32)
    second = deterministic_embedding("fixed text", 32)
    assert first == second
    assert math.sqrt(sum(value * value for value in first)) == pytest.approx(1.0)


def test_upsert_search_delete_and_exact_filtering(tmp_path: object) -> None:
    store = EmulatorStore(f"{tmp_path}/vectors.sqlite3")
    assert store.upsert([datapoint()]) == 1
    assert store.upsert([datapoint()]) == 1

    results = store.search([1.0, 0.0], restrictions(), 8)
    assert [item["vector_id"] for item in results] == ["vector-1"]
    assert store.search([1.0, 0.0], restrictions(tenant_id="other-tenant"), 8) == []
    assert store.search([1.0, 0.0], restrictions(active_generations=["8:handbook:2"]), 8) == []

    assert store.delete(["vector-1"]) == 1
    assert store.delete(["vector-1"]) == 0
    assert store.search([1.0, 0.0], restrictions(), 8) == []


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"tenant_id": "dev-tenant"},
        {
            "tenant_id": "dev-tenant",
            "corpus": "engineering",
            "index_version": "index-1",
            "active_generations": [],
        },
    ],
)
def test_search_rejects_missing_mandatory_restrictions(
    tmp_path: object, value: dict[str, object]
) -> None:
    store = EmulatorStore(f"{tmp_path}/vectors.sqlite3")
    with pytest.raises(ApiError, match="mandatory"):
        store.search([1.0], value, 1)


def test_failure_injection_is_bounded_and_reset_clears_state(tmp_path: object) -> None:
    store = EmulatorStore(f"{tmp_path}/vectors.sqlite3")
    store.inject_failure("search", 2, "DEPENDENCY_UNAVAILABLE")
    with pytest.raises(ApiError, match="injected"):
        store.maybe_fail("search")
    with pytest.raises(ApiError, match="injected"):
        store.maybe_fail("search")
    store.maybe_fail("search")

    store.upsert([datapoint()])
    store.reset()
    assert store.search([1.0, 0.0], restrictions(), 1) == []


def test_upsert_rejects_malformed_active_generation(tmp_path: object) -> None:
    store = EmulatorStore(f"{tmp_path}/vectors.sqlite3")
    value = datapoint()
    value["active_generation"] = "handbook:1"
    with pytest.raises(ApiError, match="invalid datapoint"):
        store.upsert([value])
