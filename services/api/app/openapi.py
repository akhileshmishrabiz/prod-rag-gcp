"""Deterministic OpenAPI generation helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


def install_openapi(app: FastAPI) -> None:
    def schema() -> dict[str, Any]:
        if app.openapi_schema is None:
            generated = get_openapi(
                title=app.title,
                version=app.version,
                description=(
                    "Fixed-tenant, IAM-authenticated development API. "
                    "The tenant is server configured and never caller selectable."
                ),
                routes=app.routes,
            )
            generated["openapi"] = "3.1.0"
            components = generated.setdefault("components", {})
            components.setdefault("securitySchemes", {})["googleCloudIam"] = {
                "type": "openIdConnect",
                "openIdConnectUrl": (
                    "https://accounts.google.com/.well-known/openid-configuration"
                ),
                "description": (
                    "Cloud Run IAM identity token; enforced before application routing."
                ),
            }
            generated["security"] = [{"googleCloudIam": []}]
            for path in ("/healthz", "/readyz"):
                generated["paths"][path]["get"]["security"] = []
            app.openapi_schema = generated
        return app.openapi_schema

    app.openapi = schema  # type: ignore[method-assign]


def export_openapi(app: FastAPI, destination: Path) -> None:
    """Write the generated OpenAPI schema as stable JSON."""
    destination.write_text(
        json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def semantic_surface(document: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the public wire shape used for YAML/generated drift checks."""
    operations: dict[str, Any] = {}
    for path, method, success in (
        ("/healthz", "get", "200"),
        ("/readyz", "get", "200"),
        ("/v1/documents", "post", "202"),
        ("/v1/documents/{document_id}", "get", "200"),
        ("/v1/documents/{document_id}", "delete", "202"),
        ("/v1/query", "post", "200"),
    ):
        operation = document["paths"][path][method]
        request_shape = None
        if "requestBody" in operation:
            request_schema = operation["requestBody"]["content"]["application/json"]["schema"]
            request_shape = _schema_shape(document, request_schema)
        response_schema = operation["responses"][success]["content"]["application/json"]["schema"]
        operations[f"{method.upper()} {path}"] = {
            "operationId": operation["operationId"],
            "security": operation.get("security", document.get("security")),
            "request": request_shape,
            "response": _schema_shape(document, response_schema),
        }
    return {"operations": operations}


def _schema_shape(document: Mapping[str, Any], raw: Mapping[str, Any]) -> dict[str, Any]:
    schema = _resolve(document, raw)
    properties = schema.get("properties", {})
    fields: dict[str, Any] = {}
    for name, value in properties.items():
        resolved = _resolve(document, value)
        field: dict[str, Any] = {}
        if "default" in value:
            field["default"] = value["default"]
        elif "default" in resolved:
            field["default"] = resolved["default"]
        if resolved.get("type") == "array" and "items" in resolved:
            field["items"] = _schema_shape(document, resolved["items"])
        fields[name] = field
    return {
        "required": sorted(schema.get("required", [])),
        "fields": fields,
        "additionalProperties": schema.get("additionalProperties"),
    }


def _resolve(document: Mapping[str, Any], raw: Mapping[str, Any]) -> dict[str, Any]:
    if "$ref" not in raw:
        return dict(raw)
    target: Any = document
    for part in raw["$ref"].removeprefix("#/").split("/"):
        target = target[part]
    merged = dict(target)
    merged.update({key: value for key, value in raw.items() if key != "$ref"})
    return merged
