from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from dataclasses import dataclass


@dataclass
class RetrievedDoc:
    source: str
    text: str
    score: float

from legal_assistant.runtime.graph import build_agent_graph
from legal_assistant.runtime.nodes import RuntimeDeps
from legal_assistant.tools.base import WeatherResult


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
    manager.save_turn.return_value = None
    return manager


@pytest.mark.asyncio
async def test_graph_routes_legal_intent(
    mock_retriever,
    mock_llm,
    mock_weather_adapter,
    mock_memory_manager,
):
    deps = RuntimeDeps(
        llm=mock_llm,
        retriever=mock_retriever,
        weather_adapter=mock_weather_adapter,
        memory_manager=mock_memory_manager,
    )
    graph = build_agent_graph(deps)

    result = await graph.ainvoke(
        {
            "session_id": "session-legal-1",
            "messages": [{"role": "user", "content": "劳动合同试用期最长多久？"}],
        }
    )

    assert result["intent"] == "legal"
    assert result["answer"] is not None
    assert "六个月" in result["answer"] or "试用期" in result["answer"]
    assert result["citations"]
    assert result["citations"][0]["source"] == "中华人民共和国劳动法.md"
    assert result["retrieved_docs"]
    mock_retriever.retrieve.assert_called_once_with("劳动合同试用期最长多久？")
    mock_llm.ainvoke.assert_called_once()
    mock_memory_manager.save_turn.assert_awaited_once()
    save_kwargs = mock_memory_manager.save_turn.await_args.kwargs
    assert save_kwargs["session_id"] == "session-legal-1"
    assert save_kwargs["intent"] == "legal"


@pytest.mark.asyncio
async def test_graph_routes_weather_intent(
    mock_retriever,
    mock_llm,
    mock_weather_adapter,
    mock_memory_manager,
):
    mock_llm.ainvoke.return_value = AIMessage(content="北京今天晴，22度。")

    deps = RuntimeDeps(
        llm=mock_llm,
        retriever=mock_retriever,
        weather_adapter=mock_weather_adapter,
        memory_manager=mock_memory_manager,
    )
    graph = build_agent_graph(deps)

    result = await graph.ainvoke(
        {
            "session_id": "session-weather-1",
            "messages": [{"role": "user", "content": "北京今天天气怎么样？"}],
        }
    )

    assert result["intent"] == "weather"
    assert result["location"] == "北京"
    assert result["tool_result"] is not None
    assert result["tool_result"].location == "北京"
    mock_weather_adapter.get_weather.assert_awaited_once_with("北京")
    mock_retriever.retrieve.assert_not_called()


@pytest.mark.asyncio
async def test_graph_routes_general_intent(
    mock_retriever,
    mock_llm,
    mock_weather_adapter,
    mock_memory_manager,
):
    mock_llm.ainvoke.return_value = AIMessage(content="你好，我是法律助手。")

    deps = RuntimeDeps(
        llm=mock_llm,
        retriever=mock_retriever,
        weather_adapter=mock_weather_adapter,
        memory_manager=mock_memory_manager,
    )
    graph = build_agent_graph(deps)

    result = await graph.ainvoke(
        {
            "session_id": "session-general-1",
            "messages": [{"role": "user", "content": "你好"}],
        }
    )

    assert result["intent"] == "general"
    assert result["answer"] == "你好，我是法律助手。"
    mock_retriever.retrieve.assert_not_called()
    mock_weather_adapter.get_weather.assert_not_called()
