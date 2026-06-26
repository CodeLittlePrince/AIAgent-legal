from __future__ import annotations

import functools
import inspect
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

from legal_assistant.observability.langfuse_client import get_langfuse

F = TypeVar("F", bound=Callable[..., Any])


def _new_trace_id() -> str:
    return uuid.uuid4().hex


@contextmanager
def trace_chat(session_id: str) -> Iterator[str]:
    trace_id = _new_trace_id()
    client = get_langfuse()
    if client is None:
        yield trace_id
        return

    with client.start_as_current_observation(
        name="graph.total",
        as_type="span",
        trace_context={"trace_id": trace_id},
    ) as observation:
        observation.update_trace(session_id=session_id, name="chat")
        try:
            yield trace_id
        finally:
            client.flush()


class span:
    def __init__(self, name: str) -> None:
        self.name = name
        self._client = get_langfuse()
        self._observation = None

    def __enter__(self) -> None:
        if self._client is None:
            return None
        self._context = self._client.start_as_current_observation(
            name=self.name,
            as_type="span",
        )
        self._observation = self._context.__enter__()
        return None

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._client is None or self._observation is None:
            return None
        return self._context.__exit__(exc_type, exc, tb)

    def __call__(self, func: F) -> F:
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with span(self.name):
                    return await func(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with span(self.name):
                return func(*args, **kwargs)

        return sync_wrapper  # type: ignore[return-value]
