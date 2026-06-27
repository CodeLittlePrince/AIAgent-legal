"""可观测性模块的公共入口。

统一导出 Langfuse 追踪、Prometheus 指标及相关辅助函数，
供 API 层与 runtime 在记录请求、延迟、工具调用时使用。
"""

from legal_assistant.observability.langfuse_client import get_langfuse, get_langfuse_client, reset_langfuse_client
from legal_assistant.observability.metrics import (
    CHAT_LATENCY_SECONDS,
    CHAT_REQUESTS_TOTAL,
    LLM_TOKENS_TOTAL,
    TOOL_CALLS_TOTAL,
    get_metrics_content,
    record_chat_latency,
    record_chat_request,
    record_tool_call,
)
from legal_assistant.observability.langchain_tracing import (
    bind_langchain_handler,
    get_langchain_callbacks,
    langchain_invoke_config,
    reset_langchain_handler,
)
from legal_assistant.observability.tracing import span, trace_chat, update_trace_output

__all__ = [
    "CHAT_LATENCY_SECONDS",
    "CHAT_REQUESTS_TOTAL",
    "LLM_TOKENS_TOTAL",
    "TOOL_CALLS_TOTAL",
    "get_langfuse",
    "get_langfuse_client",
    "get_metrics_content",
    "record_chat_latency",
    "record_chat_request",
    "record_tool_call",
    "bind_langchain_handler",
    "get_langchain_callbacks",
    "langchain_invoke_config",
    "reset_langchain_handler",
    "reset_langfuse_client",
    "span",
    "trace_chat",
    "update_trace_output",
]
