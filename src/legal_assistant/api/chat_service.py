"""聊天业务逻辑模块。

封装同步聊天与 SSE 流式聊天的核心流程：
加载会话历史 → 调用 Agent 图 → 记录指标与追踪 → 格式化响应。
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from legal_assistant.api.schemas import Citation
from legal_assistant.config import settings
from legal_assistant.observability.langfuse_client import get_langfuse_client
from legal_assistant.observability.langchain_tracing import langchain_invoke_config
from legal_assistant.observability.metrics import (
    record_chat_latency,
    record_chat_request,
    record_tool_call,
)
from legal_assistant.observability.tracing import trace_chat, update_trace_output

# 流式接口中，根据识别出的意图向前端推送不同的「处理中」提示文案
STATUS_BY_INTENT = {
    "legal": "正在检索法律文档并生成回答…",
    "weather": "正在查询天气信息…",
    "general": "正在生成回答…",
}


@dataclass
class ChatExecution:
    """一次聊天执行的完整结果（成功或失败）。

    Attributes:
        session_id: 会话 ID。
        trace_id: Langfuse 链路追踪 ID。
        intent: 识别出的意图。
        answer: 助手回答正文。
        citations: 法律引用列表。
        disclaimer: 免责声明（仅 legal 意图）。
        error: 若执行失败，存放错误信息；成功时为 None。
    """

    session_id: str
    trace_id: str
    intent: str
    answer: str
    citations: list[Citation]
    disclaimer: str | None
    error: str | None = None


def chunk_text(text: str, *, size: int = 2) -> list[str]:
    """将文本按固定字符数切分为小块，用于模拟打字机流式输出。

    Args:
        text: 待切分的完整回答。
        size: 每块的字符数，默认 2（中文约一个字一个词组）。

    Returns:
        文本块列表；空字符串返回空列表。
    """
    if not text:
        return []
    return [text[index : index + size] for index in range(0, len(text), size)]


def format_sse(event: str, data: dict[str, Any]) -> str:
    """格式化为 Server-Sent Events (SSE) 协议要求的字符串。

    SSE 格式：event 行 + data 行 + 空行。

    Args:
        event: 事件类型名（如 delta、done、error）。
        data: 要序列化为 JSON 的载荷字典。

    Returns:
        符合 SSE 规范的完整事件字符串。
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def strip_trailing_disclaimer(answer: str, disclaimer: str | None) -> str:
    """从回答末尾移除免责声明，避免流式输出时重复展示。

    Agent 生成的 answer 可能已包含 disclaimer 文本；
    流式接口会单独发送 disclaimer 事件，因此正文部分需先剥离。

    Args:
        answer: 原始回答。
        disclaimer: 免责声明文案；为 None 时仅做 rstrip。

    Returns:
        去除末尾 disclaimer 后的回答正文。
    """
    if not disclaimer:
        return answer.rstrip()
    trimmed = answer.rstrip()
    disclaimer_text = disclaimer.strip()
    if trimmed.endswith(disclaimer_text):
        return trimmed[: -len(disclaimer_text)].rstrip()
    return trimmed


async def execute_chat(
    *,
    memory_manager: Any,
    graph: Any,
    message: str,
    session_id: str | None = None,
) -> ChatExecution:
    """执行一次完整聊天：加载历史、调用 Agent、持久化并返回结果。

    Args:
        memory_manager: 会话记忆管理器。
        graph: LangGraph 编译后的 Agent 图。
        message: 用户当前消息。
        session_id: 可选会话 ID；不传则自动生成 UUID。

    Returns:
        ChatExecution，包含回答或 error 字段。
    """
    resolved_session_id = session_id or str(uuid.uuid4())
    started_at = time.perf_counter()
    intent = "general"
    status = "success"

    # trace_chat 创建根 span、绑定 LangChain CallbackHandler，并返回 trace_id
    with trace_chat(resolved_session_id, user_message=message) as trace_id:
        history = await memory_manager.load(resolved_session_id)
        messages = [*history, {"role": "user", "content": message}]
        result = await graph.ainvoke(
            {
                "session_id": resolved_session_id,
                "messages": messages,
            },
            config=langchain_invoke_config(),
        )

        intent = result.get("intent") or "general"
        answer = result.get("answer")
        error = result.get("error")

        # 若调用了天气等外部工具，记录工具调用指标
        if result.get("tool_result") is not None:
            tool_status = "success" if error is None else "error"
            record_tool_call("weather", tool_status)

        if error and not answer:
            record_chat_request(intent, "error")
            record_chat_latency(time.perf_counter() - started_at)
            update_trace_output(intent=intent, error=error)
            return ChatExecution(
                session_id=resolved_session_id,
                trace_id=trace_id,
                intent=intent,
                answer="",
                citations=[],
                disclaimer=None,
                error=error,
            )

        if not answer:
            record_chat_request(intent, "error")
            record_chat_latency(time.perf_counter() - started_at)
            update_trace_output(intent=intent, error=error or "No answer generated")
            return ChatExecution(
                session_id=resolved_session_id,
                trace_id=trace_id,
                intent=intent,
                answer="",
                citations=[],
                disclaimer=None,
                error="No answer generated",
            )

        citations = [Citation(**item) for item in (result.get("citations") or [])]
        # 仅法律咨询类回答附加免责声明
        disclaimer = settings.legal_disclaimer if intent == "legal" else None

        record_chat_request(intent, status)
        record_chat_latency(time.perf_counter() - started_at)

        update_trace_output(
            intent=intent,
            answer=answer,
            citations_count=len(citations),
            error=error,
        )
        langfuse = get_langfuse_client()
        if langfuse is not None:
            langfuse.flush()

        return ChatExecution(
            session_id=resolved_session_id,
            trace_id=trace_id,
            intent=intent,
            answer=answer,
            citations=citations,
            disclaimer=disclaimer,
        )


async def iter_chat_sse_events(
    execution: ChatExecution,
    *,
    chunk_delay_seconds: float = 0.012,
) -> AsyncIterator[str]:
    """将 ChatExecution 转换为 SSE 事件流（不含前置 status 事件）。

    事件顺序：session → intent → citations（若有）→ delta（逐块）→ disclaimer（若有）→ done。

    Args:
        execution: 已完成的聊天执行结果。
        chunk_delay_seconds: 每块 delta 之间的延迟（秒），营造打字效果。

    Yields:
        SSE 格式字符串。
    """
    if execution.error:
        yield format_sse("error", {"message": execution.error})
        return

    yield format_sse(
        "session",
        {"session_id": execution.session_id, "trace_id": execution.trace_id},
    )
    yield format_sse("intent", {"intent": execution.intent})

    if execution.citations:
        yield format_sse(
            "citations",
            {"citations": [item.model_dump() for item in execution.citations]},
        )

    answer_body = strip_trailing_disclaimer(execution.answer, execution.disclaimer)
    for piece in chunk_text(answer_body):
        yield format_sse("delta", {"content": piece})
        if chunk_delay_seconds > 0:
            await asyncio.sleep(chunk_delay_seconds)

    if execution.disclaimer:
        yield format_sse("disclaimer", {"disclaimer": execution.disclaimer})

    yield format_sse("done", {})


async def stream_chat_events(
    *,
    memory_manager: Any,
    graph: Any,
    message: str,
    session_id: str | None = None,
    chunk_delay_seconds: float = 0.012,
) -> AsyncIterator[str]:
    """流式聊天的完整入口：先推送状态，再执行聊天并流式返回结果。

    Args:
        memory_manager: 会话记忆管理器。
        graph: Agent 图。
        message: 用户消息。
        session_id: 可选会话 ID。
        chunk_delay_seconds: delta 块之间的延迟。

    Yields:
        SSE 格式字符串，包括 status、error、delta、done 等事件。
    """
    yield format_sse("status", {"message": "正在理解您的问题…"})

    try:
        execution = await execute_chat(
            memory_manager=memory_manager,
            graph=graph,
            message=message,
            session_id=session_id,
        )
    except Exception as exc:
        record_chat_request("unknown", "error")
        yield format_sse("error", {"message": f"Chat failed: {exc}"})
        return

    if execution.error:
        yield format_sse("error", {"message": execution.error})
        return

    yield format_sse(
        "status",
        {"message": STATUS_BY_INTENT.get(execution.intent, "正在生成回答…")},
    )

    async for event in iter_chat_sse_events(
        execution,
        chunk_delay_seconds=chunk_delay_seconds,
    ):
        yield event
