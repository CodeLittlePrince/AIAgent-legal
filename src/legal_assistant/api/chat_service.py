"""聊天业务逻辑模块。"""

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

from legal_assistant.tools.constants import LEGAL_SEARCH_TOOL, WEATHER_FORECAST_TOOL

LEGAL_TOOL = LEGAL_SEARCH_TOOL
WEATHER_TOOL = WEATHER_FORECAST_TOOL

STATUS_BY_TOOLS = {
    LEGAL_TOOL: "正在检索法律文档并生成回答…",
    WEATHER_TOOL: "正在查询天气信息…",
}


def _status_message(tools_used: list[str]) -> str:
    for tool in (LEGAL_TOOL, WEATHER_TOOL):
        if tool in tools_used:
            return STATUS_BY_TOOLS[tool]
    return "正在生成回答…"


def _metrics_label(tools_used: list[str]) -> str:
    if not tools_used:
        return "none"
    return "+".join(sorted(tools_used))


@dataclass
class ChatExecution:
    session_id: str
    trace_id: str
    intent: str | None
    tools_used: list[str]
    answer: str
    citations: list[Citation]
    disclaimer: str | None
    error: str | None = None


def chunk_text(text: str, *, size: int = 2) -> list[str]:
    if not text:
        return []
    return [text[index : index + size] for index in range(0, len(text), size)]


def format_sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def strip_trailing_disclaimer(answer: str, disclaimer: str | None) -> str:
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
    resolved_session_id = session_id or str(uuid.uuid4())
    started_at = time.perf_counter()

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

        tools_used = list(result.get("tools_used") or [])
        intent = result.get("intent")
        answer = result.get("answer")
        error = result.get("error")

        if WEATHER_TOOL in tools_used:
            tool_status = "success" if error is None else "error"
            record_tool_call("weather", tool_status)
        if LEGAL_TOOL in tools_used:
            tool_status = "success" if error is None else "error"
            record_tool_call("legal_search", tool_status)

        metrics_label = _metrics_label(tools_used)

        if error and not answer:
            record_chat_request(metrics_label, "error")
            record_chat_latency(time.perf_counter() - started_at)
            update_trace_output(tools_used=tools_used, error=error)
            return ChatExecution(
                session_id=resolved_session_id,
                trace_id=trace_id,
                intent=intent,
                tools_used=tools_used,
                answer="",
                citations=[],
                disclaimer=None,
                error=error,
            )

        if not answer:
            record_chat_request(metrics_label, "error")
            record_chat_latency(time.perf_counter() - started_at)
            update_trace_output(tools_used=tools_used, error=error or "No answer generated")
            return ChatExecution(
                session_id=resolved_session_id,
                trace_id=trace_id,
                intent=intent,
                tools_used=tools_used,
                answer="",
                citations=[],
                disclaimer=None,
                error="No answer generated",
            )

        citations = [Citation(**item) for item in (result.get("citations") or [])]
        disclaimer = settings.legal_disclaimer if LEGAL_TOOL in tools_used else None

        record_chat_request(metrics_label, "success")
        record_chat_latency(time.perf_counter() - started_at)

        update_trace_output(
            tools_used=tools_used,
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
            tools_used=tools_used,
            answer=answer,
            citations=citations,
            disclaimer=disclaimer,
        )


async def iter_chat_sse_events(
    execution: ChatExecution,
    *,
    chunk_delay_seconds: float = 0.012,
) -> AsyncIterator[str]:
    if execution.error:
        yield format_sse("error", {"message": execution.error})
        return

    yield format_sse(
        "session",
        {"session_id": execution.session_id, "trace_id": execution.trace_id},
    )
    yield format_sse("tools", {"tools_used": execution.tools_used})
    if execution.intent:
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

    yield format_sse("status", {"message": _status_message(execution.tools_used)})

    async for event in iter_chat_sse_events(
        execution,
        chunk_delay_seconds=chunk_delay_seconds,
    ):
        yield event
