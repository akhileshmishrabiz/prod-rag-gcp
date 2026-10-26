"""FastAPI process entry point."""

from collections.abc import Callable

from fastapi import FastAPI

from packages.rag_core.config import RuntimeSettings
from packages.rag_core.errors import ConfigurationError
from services.api.app.config import load_settings
from services.api.app.dependencies import ApiDependencies
from services.api.app.errors import install_error_boundary
from services.api.app.openapi import install_openapi
from services.api.app.routes.documents import router as documents_router
from services.api.app.routes.health import router as health_router
from services.api.app.routes.query import router as query_router


def create_app(
    settings_loader: Callable[[], RuntimeSettings] = load_settings,
    dependencies: ApiDependencies | None = None,
) -> FastAPI:
    """Build the API with an injectable, fail-closed configuration loader."""
    app = FastAPI(title="Development RAG API", version="1.0.0")
    try:
        app.state.settings = settings_loader()
        app.state.configuration_error = None
    except ConfigurationError as error:
        app.state.settings = None
        app.state.configuration_error = error
    app.state.dependencies = dependencies
    install_error_boundary(app)
    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(query_router)
    install_openapi(app)
    return app


app = create_app()
