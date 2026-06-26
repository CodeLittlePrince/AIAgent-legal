from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

CHAT_REQUESTS_TOTAL = Counter(
    "chat_requests_total",
    "Total chat requests",
    ["intent", "status"],
)

CHAT_LATENCY_SECONDS = Histogram(
    "chat_latency_seconds",
    "Chat request latency in seconds",
)

LLM_TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "Total LLM tokens",
    ["model", "direction"],
)

TOOL_CALLS_TOTAL = Counter(
    "tool_calls_total",
    "Total tool calls",
    ["tool", "status"],
)

chat_requests_total = CHAT_REQUESTS_TOTAL
chat_latency_seconds = CHAT_LATENCY_SECONDS
llm_tokens_total = LLM_TOKENS_TOTAL
tool_calls_total = TOOL_CALLS_TOTAL


def record_chat_request(intent: str, status: str) -> None:
    CHAT_REQUESTS_TOTAL.labels(intent=intent, status=status).inc()


def record_chat_latency(seconds: float) -> None:
    CHAT_LATENCY_SECONDS.observe(seconds)


def record_llm_tokens(model: str, direction: str, count: int) -> None:
    if count > 0:
        LLM_TOKENS_TOTAL.labels(model=model, direction=direction).inc(count)


def record_tool_call(tool: str, status: str) -> None:
    TOOL_CALLS_TOTAL.labels(tool=tool, status=status).inc()


def get_metrics_content() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
