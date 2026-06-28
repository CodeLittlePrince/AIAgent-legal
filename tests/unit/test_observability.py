import pytest

from legal_assistant.config import settings
from legal_assistant.observability.langfuse_client import get_langfuse, reset_langfuse_client
from legal_assistant.observability.metrics import (
    CHAT_LATENCY_SECONDS,
    CHAT_REQUESTS_TOTAL,
    LLM_TOKENS_TOTAL,
    TOOL_CALLS_TOTAL,
)
from legal_assistant.observability.langchain_tracing import (
    bind_langchain_handler,
    langchain_invoke_config,
    reset_langchain_handler,
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


def test_get_langfuse_ignores_socks_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "langfuse_enabled", True)
    monkeypatch.setattr(settings, "langfuse_public_key", "pk-test")
    monkeypatch.setattr(settings, "langfuse_secret_key", "sk-test")
    monkeypatch.setattr(settings, "langfuse_host", "http://127.0.0.1:9")
    monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:7890")
    reset_langfuse_client()

    client = get_langfuse()

    assert client is not None


def test_trace_chat_noop_when_langfuse_disabled() -> None:
    with trace_chat("session-1") as trace_id:
        assert isinstance(trace_id, str)
        assert trace_id


def test_langchain_invoke_config_empty_when_disabled() -> None:
    assert langchain_invoke_config() == {}


def test_langchain_handler_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "langfuse_enabled", True)
    monkeypatch.setattr(settings, "langfuse_public_key", "pk-test")
    monkeypatch.setattr(settings, "langfuse_secret_key", "sk-test")
    monkeypatch.setattr(settings, "langfuse_host", "http://127.0.0.1:9")
    monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:7890")
    reset_langfuse_client()

    handler = bind_langchain_handler()
    assert handler is not None
    assert langchain_invoke_config()["callbacks"] == [handler]
    reset_langchain_handler()
    assert langchain_invoke_config() == {}


def test_trace_chat_with_langfuse_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "langfuse_enabled", True)
    monkeypatch.setattr(settings, "langfuse_public_key", "pk-test")
    monkeypatch.setattr(settings, "langfuse_secret_key", "sk-test")
    monkeypatch.setattr(settings, "langfuse_host", "http://127.0.0.1:9")
    monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:7890")
    reset_langfuse_client()

    with trace_chat("session-langfuse-1", user_message="你好") as trace_id:
        assert isinstance(trace_id, str)
        assert trace_id
        assert langchain_invoke_config() != {}
    assert langchain_invoke_config() == {}


def test_span_context_manager_noop_when_langfuse_disabled() -> None:
    with span("agent.tool_loop"):
        pass


def test_span_decorator_noop_when_langfuse_disabled() -> None:
    @span("agent.tool_loop")
    def run_agent() -> str:
        return "ok"

    assert run_agent() == "ok"


@pytest.mark.asyncio
async def test_span_decorator_async_noop_when_langfuse_disabled() -> None:
    @span("llm.generate")
    async def generate() -> str:
        return "ok"

    assert await generate() == "ok"


def test_prometheus_metrics_defined() -> None:
    CHAT_REQUESTS_TOTAL.labels(tools="search_legal_knowledge", status="success").inc()
    CHAT_LATENCY_SECONDS.observe(0.12)
    LLM_TOKENS_TOTAL.labels(model="deepseek-chat", direction="input").inc(42)
    TOOL_CALLS_TOTAL.labels(tool="weather", status="success").inc()
