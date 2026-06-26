from __future__ import annotations

from typing import TYPE_CHECKING

from legal_assistant.config import settings

if TYPE_CHECKING:
    from langfuse import Langfuse

_client: Langfuse | None = None
_initialized = False


def reset_langfuse_client() -> None:
    global _client, _initialized
    _client = None
    _initialized = False


def get_langfuse() -> Langfuse | None:
    global _client, _initialized

    if not settings.langfuse_enabled:
        return None

    if _initialized:
        return _client

    _initialized = True
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return None

    from langfuse import Langfuse

    _client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
    return _client


def get_langfuse_client() -> Langfuse | None:
    return get_langfuse()
