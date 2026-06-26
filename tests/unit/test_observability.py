import pytest

from legal_assistant.config import settings
from legal_assistant.observability.langfuse_client import get_langfuse, reset_langfuse_client
from legal_assistant.observability.metrics import (
    CHAT_LATENCY_SECONDS,
    CHAT_REQUESTS_TOTAL,
    LLM_TOKENS_TOTAL,
    TOOL_CALLS_TOTAL,
)
from legal_assistant.observability.tracing import span, trace_chat


@pytest.fixture(autouse=True)
def _disable_langfuse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "langfuse_enabled", False)
    reset_langfuse_client()
    yield
    reset_langfuse_client()


def test_get_langfuse_returns_none_when_disabled() -> None:
    assert get_langfuse() is None


def test_trace_chat_noop_when_langfuse_disabled() -> None:
    with trace_chat("session-1") as trace_id:
        assert isinstance(trace_id, str)
        assert trace_id


def test_span_context_manager_noop_when_langfuse_disabled() -> None:
    with span("planner"):
        pass


def test_span_decorator_noop_when_langfuse_disabled() -> None:
    @span("planner")
    def classify() -> str:
        return "legal"

    assert classify() == "legal"


@pytest.mark.asyncio
async def test_span_decorator_async_noop_when_langfuse_disabled() -> None:
    @span("llm.generate")
    async def generate() -> str:
        return "ok"

    assert await generate() == "ok"


def test_prometheus_metrics_defined() -> None:
    CHAT_REQUESTS_TOTAL.labels(intent="legal", status="success").inc()
    CHAT_LATENCY_SECONDS.observe(0.12)
    LLM_TOKENS_TOTAL.labels(model="deepseek-chat", direction="input").inc(42)
    TOOL_CALLS_TOTAL.labels(tool="weather", status="success").inc()
