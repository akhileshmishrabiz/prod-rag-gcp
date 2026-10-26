"""Private worker process entry point."""

from collections.abc import Callable

from fastapi import FastAPI

from packages.rag_core.config import RuntimeSettings
from packages.rag_core.errors import ConfigurationError
from services.worker.app.config import load_settings
from services.worker.app.dependencies import WorkerDependencies
from services.worker.app.probes import router as probe_router
from services.worker.app.routes import router as ingestion_router


def create_app(
    settings_loader: Callable[[], RuntimeSettings] = load_settings,
    dependencies: WorkerDependencies | None = None,
) -> FastAPI:
    """Build the worker with injectable configuration and business ports."""
    app = FastAPI(title="Development RAG Worker", version="1.0.0")
    try:
        app.state.settings = settings_loader()
        app.state.configuration_error = None
    except ConfigurationError as error:
        app.state.settings = None
        app.state.configuration_error = error
    app.state.dependencies = dependencies
    app.include_router(probe_router)
    app.include_router(ingestion_router)
    return app


app = create_app()
