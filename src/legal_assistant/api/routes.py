from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse

from legal_assistant.api.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    FeedbackRequest,
    SessionResponse,
)
from legal_assistant.config import settings
from legal_assistant.memory.manager import MemoryManager
from legal_assistant.observability.langfuse_client import get_langfuse_client
from legal_assistant.observability.metrics import (
    get_metrics_content,
    record_chat_latency,
    record_chat_request,
    record_tool_call,
)
from legal_assistant.observability.tracing import trace_chat
from legal_assistant.runtime.nodes import RuntimeDeps

router = APIRouter()


@dataclass
class AppServices:
    memory_manager: MemoryManager
    runtime_deps: RuntimeDeps
    graph: Any


def get_services(request: Request) -> AppServices:
    services = getattr(request.app.state, "services", None)
    if services is None:
        raise HTTPException(status_code=503, detail="Application services not initialized")
    return services


@router.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request_body: ChatRequest, request: Request) -> ChatResponse:
    services = get_services(request)
    session_id = request_body.session_id or str(uuid.uuid4())
    started_at = time.perf_counter()
    intent = "general"
    status = "success"

    with trace_chat(session_id) as trace_id:
        try:
            history = await services.memory_manager.load(session_id)
        except Exception as exc:
            record_chat_request("unknown", "error")
            raise HTTPException(status_code=503, detail=f"Memory unavailable: {exc}") from exc

        messages = [*history, {"role": "user", "content": request_body.message}]
        try:
            result = await services.graph.ainvoke(
                {
                    "session_id": session_id,
                    "messages": messages,
                }
            )
        except Exception as exc:
            record_chat_request("unknown", "error")
            raise HTTPException(status_code=503, detail=f"Agent execution failed: {exc}") from exc

        intent = result.get("intent") or "general"
        answer = result.get("answer")
        error = result.get("error")

        if result.get("tool_result") is not None:
            tool_status = "success" if error is None else "error"
            record_tool_call("weather", tool_status)

        if error and not answer:
            status = "error"
            record_chat_request(intent, status)
            raise HTTPException(status_code=503, detail=error)

        if not answer:
            status = "error"
            record_chat_request(intent, status)
            raise HTTPException(status_code=503, detail="No answer generated")

        citations = [Citation(**item) for item in (result.get("citations") or [])]
        disclaimer = settings.legal_disclaimer if intent == "legal" else None

        record_chat_request(intent, status)
        record_chat_latency(time.perf_counter() - started_at)

        return ChatResponse(
            session_id=session_id,
            intent=intent,
            answer=answer,
            citations=citations,
            disclaimer=disclaimer,
            trace_id=trace_id,
        )


@router.get("/api/v1/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str, request: Request) -> SessionResponse:
    services = get_services(request)
    try:
        messages = await services.memory_manager.load(session_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Memory unavailable: {exc}") from exc
    return SessionResponse(session_id=session_id, messages=messages)


@router.delete("/api/v1/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str, request: Request) -> Response:
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
    from legal_assistant.knowledge.ingest import ingest_legal_documents

    try:
        indexed = ingest_legal_documents()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Reindex failed: {exc}") from exc
    return {"status": "ok", "indexed_nodes": indexed}


@router.post("/api/v1/feedback")
async def submit_feedback(request_body: FeedbackRequest) -> dict[str, str]:
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
        client.flush()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Feedback submission failed: {exc}") from exc

    return {"status": "ok"}


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
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

    overall = "ok" if all(value == "ok" or value == "disabled" for value in checks.values()) else "degraded"
    return {"status": overall, "checks": checks}


@router.get("/metrics")
async def metrics() -> PlainTextResponse:
    content, content_type = get_metrics_content()
    return PlainTextResponse(content=content.decode("utf-8"), media_type=content_type)
