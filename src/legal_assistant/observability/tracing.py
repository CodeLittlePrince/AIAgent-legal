"""基于 Langfuse 的分布式追踪辅助工具。

提供：
- ``trace_chat``：包裹整次聊天请求的根 span，并绑定 session_id
- ``span``：可作为上下文管理器或函数装饰器，记录子步骤耗时与层级
- ``update_trace_output``：在请求结束时写入 trace 输出摘要
"""

from __future__ import annotations

import functools
import inspect
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

from langfuse import propagate_attributes

from legal_assistant.observability.langchain_tracing import (
    bind_langchain_handler,
    reset_langchain_handler,
)
from legal_assistant.observability.langfuse_client import get_langfuse

F = TypeVar("F", bound=Callable[..., Any])

_TRACE_OUTPUT_PREVIEW_LEN = 500


def _new_trace_id() -> str:
    """生成新的 trace 标识（32 位十六进制 UUID，无连字符）。"""
    return uuid.uuid4().hex


def _string_metadata(**fields: Any) -> dict[str, str]:
    """将字段转为 Langfuse metadata 要求的 ``dict[str, str]``。"""
    return {key: str(value) for key, value in fields.items() if value is not None}


def update_trace_output(
    *,
    intent: str,
    answer: str | None = None,
    citations_count: int = 0,
    error: str | None = None,
) -> None:
    """更新当前根 span 的输出与 metadata（避免记录完整长回答）。"""
    client = get_langfuse()
    if client is None:
        return

    preview = answer
    if answer and len(answer) > _TRACE_OUTPUT_PREVIEW_LEN:
        preview = answer[:_TRACE_OUTPUT_PREVIEW_LEN] + "…"

    client.update_current_span(
        output={
            "intent": intent,
            "answer_preview": preview,
            "citations_count": citations_count,
            "error": error,
        },
        metadata=_string_metadata(
            intent=intent,
            citations_count=citations_count,
            has_error=bool(error),
        ),
    )


@contextmanager
def trace_chat(session_id: str, *, user_message: str | None = None) -> Iterator[str]:
    """为一次聊天会话创建 Langfuse 根追踪上下文。

    在 Langfuse 中创建名为 ``chat-response`` 的根 span，并通过
    ``propagate_attributes`` 附加 ``session_id`` 与 ``trace_name``。
    同时绑定 LangChain ``CallbackHandler``，供图内 LLM 调用自动上报 generation。

    Args:
        session_id: 业务侧会话 ID，会写入 Langfuse 属性便于按会话筛选。
        user_message: 用户当前消息；仅写入 trace input，避免泄露其它参数。

    Yields:
        本次请求的 trace_id 字符串（无论 Langfuse 是否可用）。
    """
    trace_id = _new_trace_id()
    client = get_langfuse()
    if client is None:
        yield trace_id
        return

    bind_langchain_handler(trace_id)
    trace_input = {"message": user_message} if user_message else None
    try:
        with propagate_attributes(session_id=session_id, trace_name="chat-response"):
            with client.start_as_current_observation(
                name="chat-response",
                as_type="span",
                trace_context={"trace_id": trace_id},
                input=trace_input,
            ):
                try:
                    yield trace_id
                finally:
                    client.flush()
    finally:
        reset_langchain_handler()


class span:
    """Langfuse span 的上下文管理器，也可作为装饰器使用。

    用法示例::

        with span("retriever.search"):
            docs = retriever.retrieve(query)

        @span("llm.invoke")
        async def call_llm(...):
            ...

    当 Langfuse 客户端不可用时，进入/退出均为空操作，不影响业务代码。
    """

    def __init__(self, name: str) -> None:
        """记录 span 名称，并在构造时解析 Langfuse 客户端（可能为 None）。

        Args:
            name: 在 Langfuse UI 中显示的 span 名称，建议用 ``模块.动作`` 形式。
        """
        self.name = name
        self._client = get_langfuse()
        self._observation = None

    def __enter__(self) -> None:
        """进入同步上下文，开启当前 observation。"""
        if self._client is None:
            return None
        self._context = self._client.start_as_current_observation(
            name=self.name,
            as_type="span",
        )
        self._observation = self._context.__enter__()
        return None

    def __exit__(self, exc_type, exc, tb) -> None:
        """退出上下文，结束 observation；若有异常会传递给 Langfuse 上下文。"""
        if self._client is None or self._observation is None:
            return None
        return self._context.__exit__(exc_type, exc, tb)

    def __call__(self, func: F) -> F:
        """将 ``span`` 用作装饰器：自动包裹 sync/async 函数的执行过程。

        根据被装饰函数是否为协程函数，返回对应的 wrapper，
        在调用前后进入/退出同名 span。
        """
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
