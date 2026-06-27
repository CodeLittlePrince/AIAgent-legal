"""LangChain / LangGraph 与 Langfuse 的集成辅助。

通过 ``CallbackHandler`` 自动采集 LLM 调用的 model、token 用量与 generation 类型，
并在单次请求内用 contextvar 传递 handler，供图节点中的 ``ainvoke`` 复用。
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from legal_assistant.observability.langfuse_client import get_langfuse

_handler_var: ContextVar[Any | None] = ContextVar("langfuse_langchain_handler", default=None)


def bind_langchain_handler(trace_id: str) -> Any | None:
    """为当前 async 上下文绑定 LangChain CallbackHandler。

    Args:
        trace_id: 与 ``trace_chat`` 根 span 一致的 trace id。

    Returns:
        已绑定的 ``CallbackHandler``，Langfuse 未启用时为 ``None``。
    """
    if get_langfuse() is None:
        _handler_var.set(None)
        return None

    try:
        from langfuse.langchain import CallbackHandler
    except ModuleNotFoundError:
        _handler_var.set(None)
        return None

    handler = CallbackHandler(trace_context={"trace_id": trace_id})
    _handler_var.set(handler)
    return handler


def reset_langchain_handler() -> None:
    """清除当前上下文中的 LangChain handler（请求结束时调用）。"""
    _handler_var.set(None)


def get_langchain_callbacks() -> list[Any]:
    """返回当前请求应传给 LangChain ``ainvoke`` 的 callbacks 列表。"""
    handler = _handler_var.get()
    return [handler] if handler is not None else []


def langchain_invoke_config() -> dict[str, Any]:
    """构造 LangChain / LangGraph ``config`` 字典（含 callbacks）。"""
    callbacks = get_langchain_callbacks()
    if not callbacks:
        return {}
    return {"callbacks": callbacks}
