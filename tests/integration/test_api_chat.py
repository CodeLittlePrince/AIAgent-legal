from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from legal_assistant.main import create_app
from legal_assistant.runtime.nodes import RuntimeDeps
from legal_assistant.tools.base import WeatherResult


@dataclass
class RetrievedDoc:
    source: str
    text: str
    score: float


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        RetrievedDoc(
            source="中华人民共和国劳动法.md",
            text="劳动合同试用期最长不得超过六个月。",
            score=0.92,
        )
    ]
    return retriever


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.ainvoke.return_value = AIMessage(content="试用期最长不超过六个月。")
    return llm


@pytest.fixture
def mock_weather_adapter():
    adapter = AsyncMock()
    adapter.get_weather.return_value = WeatherResult(
        location="北京",
        temperature=22.0,
        conditions="晴",
        forecast_summary="未来三天以晴为主",
        raw_source={"provider": "mock"},
    )
    return adapter


@pytest.fixture
def mock_memory_manager():
    manager = AsyncMock()
    manager.load = AsyncMock(return_value=[])
    manager.save_turn = AsyncMock(return_value=None)

    redis_store = AsyncMock()
    redis_store.ping = AsyncMock(return_value=True)
    postgres_store = AsyncMock()
    postgres_store.ping = AsyncMock(return_value=True)

    manager.redis = redis_store
    manager.postgres = postgres_store
    return manager


@pytest.fixture
def runtime_deps(
    mock_retriever,
    mock_llm,
    mock_weather_adapter,
    mock_memory_manager,
):
    return RuntimeDeps(
        llm=mock_llm,
        retriever=mock_retriever,
        weather_adapter=mock_weather_adapter,
        memory_manager=mock_memory_manager,
    )


@pytest.fixture
def client(runtime_deps, mock_memory_manager):
    app = create_app(
        memory_manager=mock_memory_manager,
        runtime_deps=runtime_deps,
        skip_db_init=True,
        skip_auto_ingest=True,
        mount_web_ui=False,
    )
    with TestClient(app) as test_client:
        yield test_client


def test_chat_legal_returns_citations_disclaimer_and_trace(client):
    response = client.post(
        "/api/v1/chat",
        json={"session_id": None, "message": "劳动合同试用期最长多久？"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "legal"
    assert data["session_id"]
    assert data["trace_id"]
    assert data["citations"]
    assert data["citations"][0]["source"] == "中华人民共和国劳动法.md"
    assert data["disclaimer"] == "本回答仅供参考，不构成法律意见，具体问题请咨询执业律师。"
    assert "六个月" in data["answer"] or "试用期" in data["answer"]


def test_chat_weather_routes_without_citations(client):
    response = client.post(
        "/api/v1/chat",
        json={"session_id": "session-weather-1", "message": "北京今天天气怎么样？"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "weather"
    assert data["session_id"] == "session-weather-1"
    assert data["citations"] == []
    assert data["disclaimer"] is None


def test_health_ok_with_mocked_services(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert "checks" in body


def test_chat_general_routes_without_citations(client):
    response = client.post(
        "/api/v1/chat",
        json={"message": "你好，介绍一下你自己"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "general"
    assert data["citations"] == []
    assert data["disclaimer"] is None


def test_chat_persists_turn_via_graph(client, mock_memory_manager):
    response = client.post(
        "/api/v1/chat",
        json={"session_id": "session-save-1", "message": "你好"},
    )

    assert response.status_code == 200
    mock_memory_manager.save_turn.assert_awaited_once()
    call_kwargs = mock_memory_manager.save_turn.await_args.kwargs
    assert call_kwargs["session_id"] == "session-save-1"
    assert call_kwargs["user_msg"] == "你好"
    assert call_kwargs["intent"] == "general"


def test_get_session_returns_messages(client, mock_memory_manager):
    mock_memory_manager.load.return_value = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好，我是法律助手。"},
    ]

    response = client.get("/api/v1/sessions/session-1")

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "session-1"
    assert len(data["messages"]) == 2


def test_delete_session_returns_204(client, mock_memory_manager):
    mock_memory_manager.delete_session.return_value = True

    response = client.delete("/api/v1/sessions/session-1")

    assert response.status_code == 204
    mock_memory_manager.delete_session.assert_awaited_once_with("session-1")


def test_delete_session_not_found(client, mock_memory_manager):
    mock_memory_manager.delete_session.return_value = False

    response = client.delete("/api/v1/sessions/missing-session")

    assert response.status_code == 404


def test_reindex_knowledge(client):
    with patch(
        "legal_assistant.knowledge.ingest.ingest_legal_documents",
        return_value=42,
    ):
        response = client.post("/api/v1/knowledge/reindex")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "indexed_nodes": 42}


def test_feedback_skipped_when_langfuse_disabled(client):
    response = client.post(
        "/api/v1/feedback",
        json={"trace_id": "trace-123", "score": 1, "comment": "helpful"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "skipped"


def test_metrics_returns_prometheus_format(client):
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "chat_requests_total" in response.text


def test_chat_stream_returns_sse_events(client):
    with client.stream(
        "POST",
        "/api/v1/chat/stream",
        json={"message": "你好"},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    assert "event: session" in body
    assert "event: intent" in body
    assert "event: status" in body
    assert "event: delta" in body
    assert "event: done" in body
    assert "general" in body


def test_chat_stream_legal_includes_citations(client):
    with client.stream(
        "POST",
        "/api/v1/chat/stream",
        json={"message": "劳动合同试用期最长多久？"},
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    assert "event: citations" in body
    assert "event: disclaimer" in body
    assert "legal" in body
    assert body.index("event: citations") < body.index("event: delta")
