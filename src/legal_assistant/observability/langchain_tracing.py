"""LangChain / LangGraph 与 Langfuse 的集成辅助。

通过 ``CallbackHandler`` 自动采集 LLM 调用的 model、token 用量与 generation 类型，
并在单次请求内用 contextvar 传递 handler，供节点内的 ``ainvoke`` 复用。

注意：外层 ``StateGraph`` 的 ``ainvoke`` 不要传入该 handler，否则 Langfuse
会把外层编排图与内层 ``create_agent`` 图合并成一张错误的 Agent Graph。
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from langchain_core.runnables.config import RunnableConfig

from legal_assistant.observability.langfuse_client import get_langfuse

_handler_var: ContextVar[Any | None] = ContextVar("langfuse_langchain_handler", default=None)


def bind_langchain_handler() -> Any | None:
    """为当前 async 上下文绑定 LangChain CallbackHandler。

    须在 ``start_as_current_observation`` 上下文内调用，以便 LangGraph
    observation 自动继承当前 trace 并嵌套在根 span 之下。

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

    handler = CallbackHandler()
    _handler_var.set(handler)
    return handler


def reset_langchain_handler() -> None:
    """清除当前上下文中的 LangChain handler（请求结束时调用）。"""
    _handler_var.set(None)


def get_langchain_callbacks() -> list[Any]:
    """返回当前请求应传给 LangChain ``ainvoke`` 的 callbacks 列表。"""
    handler = _handler_var.get()
    return [handler] if handler is not None else []


def langchain_invoke_config() -> RunnableConfig:
    """构造 LangChain / LangGraph ``config`` 字典（含 callbacks）。"""
    callbacks = get_langchain_callbacks()
    if not callbacks:
        return {}
    return {"callbacks": callbacks}
