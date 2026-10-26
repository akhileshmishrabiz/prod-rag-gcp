"""Compose production services with local-only dependencies."""

from collections.abc import Callable

from fastapi import FastAPI

from local.emulator.wiring import build_api_dependencies, build_worker_dependencies
from packages.rag_core.config import AppMode, RuntimeSettings
from services.api.app.config import load_settings as load_api_settings
from services.api.app.main import create_app as create_api_app
from services.worker.app.config import load_settings as load_worker_settings
from services.worker.app.main import create_app as create_worker_app


def _local_settings(loader: Callable[[], RuntimeSettings]) -> RuntimeSettings:
    settings = loader()
    if settings.app_mode is not AppMode.LOCAL:
        raise RuntimeError("local runtime requires APP_MODE=local")
    return settings


def build_local_api(
    settings_loader: Callable[[], RuntimeSettings] = load_api_settings,
) -> FastAPI:
    settings = _local_settings(settings_loader)
    return create_api_app(
        settings_loader=lambda: settings,
        dependencies=build_api_dependencies(settings),
    )


def build_local_worker(
    settings_loader: Callable[[], RuntimeSettings] = load_worker_settings,
) -> FastAPI:
    settings = _local_settings(settings_loader)
    return create_worker_app(
        settings_loader=lambda: settings,
        dependencies=build_worker_dependencies(settings),
    )
