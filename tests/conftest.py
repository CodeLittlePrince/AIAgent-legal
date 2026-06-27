"""Shared pytest fixtures for all test suites."""

import pytest

from legal_assistant.config import settings
from legal_assistant.observability.langfuse_client import reset_langfuse_client


@pytest.fixture(autouse=True)
def _disable_langfuse_for_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests hermetic — do not require Langfuse credentials or network."""
    monkeypatch.setattr(settings, "langfuse_enabled", False)
    reset_langfuse_client()
    yield
    reset_langfuse_client()
