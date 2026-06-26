from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

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
