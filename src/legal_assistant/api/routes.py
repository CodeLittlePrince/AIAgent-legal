"""HTTP API 路由模块。

定义所有 REST 端点：聊天、流式聊天、会话管理、知识库重建、
用户反馈、健康检查和 Prometheus 指标。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse, StreamingResponse

from legal_assistant.api.chat_service import execute_chat, stream_chat_events
from legal_assistant.api.schemas import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    SessionResponse,
)
from legal_assistant.config import settings
from legal_assistant.memory.manager import MemoryManager
from legal_assistant.observability.langfuse_client import get_langfuse_client
from legal_assistant.observability.metrics import get_metrics_content, record_chat_request
from legal_assistant.runtime.nodes import RuntimeDeps

router = APIRouter()


@dataclass
class AppServices:
    """应用启动时注入的核心服务容器。

    保存在 app.state.services 中，供各路由处理器共享访问。

    Attributes:
        memory_manager: 会话记忆的读写（Redis + PostgreSQL）。
        runtime_deps: Agent 节点所需的运行时依赖。
        graph: 编译后的 LangGraph 状态图，执行完整对话流程。
    """

    memory_manager: MemoryManager
    runtime_deps: RuntimeDeps
    graph: Any


def get_services(request: Request) -> AppServices:
    """从 FastAPI 请求对象中获取已初始化的应用服务。

    Args:
        request: 当前 HTTP 请求。

    Returns:
        包含 memory_manager、graph 等的 AppServices 实例。

    Raises:
        HTTPException: 服务尚未初始化（503），通常表示应用仍在启动中。
    """
    services = getattr(request.app.state, "services", None)
    if services is None:
        raise HTTPException(status_code=503, detail="Application services not initialized")
    return services


@router.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request_body: ChatRequest, request: Request) -> ChatResponse:
    """同步聊天接口：等待 Agent 完整执行后一次性返回结果。

    Args:
        request_body: 包含 message 和可选 session_id 的请求体。
        request: FastAPI 请求对象，用于获取应用服务。

    Returns:
        包含回答、意图、引用和 trace_id 的 ChatResponse。
    """
    services = get_services(request)

    try:
        execution = await execute_chat(
            memory_manager=services.memory_manager,
            graph=services.graph,
            message=request_body.message,
            session_id=request_body.session_id,
        )
    except Exception as exc:
        record_chat_request("unknown", "error")
        raise HTTPException(status_code=503, detail=f"Chat failed: {exc}") from exc

    if execution.error:
        raise HTTPException(status_code=503, detail=execution.error)

    return ChatResponse(
        session_id=execution.session_id,
        intent=execution.intent,
        answer=execution.answer,
        citations=execution.citations,
        disclaimer=execution.disclaimer,
        trace_id=execution.trace_id,
    )


@router.post("/api/v1/chat/stream")
async def chat_stream(request_body: ChatRequest, request: Request) -> StreamingResponse:
    """流式聊天接口：通过 Server-Sent Events (SSE) 逐步推送回答。

    前端可实时显示「正在理解…」「正在检索…」等状态，
    并以小块（delta）形式流式展示回答正文。

    Args:
        request_body: 聊天请求体。
        request: FastAPI 请求对象。

    Returns:
        media_type 为 text/event-stream 的 StreamingResponse。
    """
    services = get_services(request)

    async def event_generator():
        """异步生成器：逐条 yield SSE 格式的事件字符串。"""
        async for event in stream_chat_events(
            memory_manager=services.memory_manager,
            graph=services.graph,
            message=request_body.message,
            session_id=request_body.session_id,
        ):
            yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",  # 禁止浏览器缓存流式响应
            "Connection": "keep-alive",  # 保持长连接
            "X-Accel-Buffering": "no",  # 告知 Nginx 不要缓冲 SSE
        },
    )


@router.get("/api/v1/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str, request: Request) -> SessionResponse:
    """获取指定会话的历史消息列表。

    Args:
        session_id: 会话唯一标识。
        request: FastAPI 请求对象。

    Returns:
        会话 ID 及 messages 列表。
    """
    services = get_services(request)
    try:
        messages = await services.memory_manager.load(session_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Memory unavailable: {exc}") from exc
    return SessionResponse(session_id=session_id, messages=messages)


@router.delete("/api/v1/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str, request: Request) -> Response:
    """删除指定会话及其全部历史消息。

    Args:
        session_id: 要删除的会话 ID。
        request: FastAPI 请求对象。

    Returns:
        成功时返回 204 No Content；会话不存在时返回 404。
    """
    services = get_services(request)
    try:
        deleted = await services.memory_manager.delete_session(session_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Memory unavailable: {exc}") from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return Response(status_code=204)


@router.post("/api/v1/knowledge/reindex")
async def reindex_knowledge() -> dict[str, int | str]:
    """手动触发法律知识库重新入库。

    适用于更新了法律文档源文件后，需要刷新向量索引的场景。

    Returns:
        包含 status 和 indexed_nodes（入库文档块数量）的字典。
    """
    from legal_assistant.knowledge.ingest import ingest_legal_documents

    try:
        indexed = ingest_legal_documents()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Reindex failed: {exc}") from exc
    return {"status": "ok", "indexed_nodes": indexed}


@router.post("/api/v1/feedback")
async def submit_feedback(request_body: FeedbackRequest) -> dict[str, str]:
    """提交用户对某次对话的评分反馈到 Langfuse。

    Args:
        request_body: 包含 trace_id、score 和可选 comment。

    Returns:
        status 为 ok；Langfuse 未启用时返回 skipped。
    """
    client = get_langfuse_client()
    if client is None:
        return {"status": "skipped", "reason": "langfuse_disabled"}

    try:
        client.create_score(
            name="user_feedback",
            value=float(request_body.score),
            trace_id=request_body.trace_id,
            comment=request_body.comment,
            data_type="NUMERIC",
        )
        client.flush()  # 确保评分立即发送到 Langfuse 服务端
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Feedback submission failed: {exc}") from exc

    return {"status": "ok"}


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    """健康检查端点：探测 Redis、PostgreSQL、Chroma、Langfuse 等依赖是否可用。

    Args:
        request: FastAPI 请求对象。

    Returns:
        overall status（ok 或 degraded）及各组件的 checks 详情。
    """
    services = get_services(request)
    checks: dict[str, str] = {}

    try:
        await services.memory_manager.redis.ping()
        checks["redis"] = "ok"
    except Exception as exc:
        checks["redis"] = f"error: {exc}"

    try:
        await services.memory_manager.postgres.ping()
        checks["postgres"] = "ok"
    except Exception as exc:
        checks["postgres"] = f"error: {exc}"

    try:
        from legal_assistant.knowledge.chroma_client import get_chroma_client

        client = get_chroma_client()
        client.heartbeat()
        checks["chroma"] = "ok"
    except Exception as exc:
        checks["chroma"] = f"error: {exc}"

    if settings.langfuse_enabled:
        langfuse_client = get_langfuse_client()
        checks["langfuse"] = "ok" if langfuse_client is not None else "error: not configured"
    else:
        checks["langfuse"] = "disabled"

    # disabled 视为正常；只有 error 状态才导致 overall 为 degraded
    overall = "ok" if all(value == "ok" or value == "disabled" for value in checks.values()) else "degraded"
    return {"status": overall, "checks": checks}


@router.get("/metrics")
async def metrics() -> PlainTextResponse:
    """Prometheus 格式的应用指标端点。

    Returns:
        纯文本格式的 counter/histogram 等指标数据。
    """
    content, content_type = get_metrics_content()
    return PlainTextResponse(content=content.decode("utf-8"), media_type=content_type)
