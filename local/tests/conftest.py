"""Reuse shared settings fixtures for local-only tests."""

from tests.conftest import local_settings, local_values

__all__ = ["local_settings", "local_values"]
