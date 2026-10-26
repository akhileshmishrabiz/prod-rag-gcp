"""Local worker entry point."""

from local.runtime.apps import build_local_worker

app = build_local_worker()
