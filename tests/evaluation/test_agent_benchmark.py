from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from legal_assistant.evaluation.agent_benchmark.metrics import compute_metrics, write_report
from legal_assistant.evaluation.agent_benchmark.runner import BenchmarkRunner
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

    async def _ainvoke(messages, *args, **kwargs):
        if isinstance(messages, list):
            last = messages[-1]
            content = getattr(last, "content", str(last))
        else:
            content = str(messages)

        if any(keyword in content for keyword in ("试用期", "劳动", "法律")):
            return AIMessage(content="试用期最长不超过六个月。")
        if any(keyword in content for keyword in ("天气", "气温", "明天", "预报")):
            location = "上海" if "上海" in content else "北京"
            if "明天" in content:
                return AIMessage(content=f"{location}明天多云，气温18到25度，适合出行。")
            return AIMessage(content=f"{location}今天晴，气温22度，未来三天以晴为主。")
        return AIMessage(content="你好，我是智能法律助手，可以解答法律、天气和一般问题。")

    llm.ainvoke.side_effect = _ainvoke
    return llm


@pytest.fixture
def mock_weather_adapter():
    adapter = AsyncMock()
    adapter.get_weather.return_value = WeatherResult(
        location="上海",
        temperature=22.0,
        conditions="晴",
        forecast_summary="未来三天以晴为主",
        raw_source={"provider": "mock"},
    )
    return adapter


@pytest.fixture
def mock_memory_manager():
    manager = AsyncMock()
    stored: dict[str, list[dict[str, str]]] = {}

    async def load(session_id: str):
        return list(stored.get(session_id, []))

    async def save_turn(session_id: str, user_msg: str, assistant_msg: str, intent: str | None = None):
        stored.setdefault(session_id, []).append({"role": "user", "content": user_msg})
        stored.setdefault(session_id, []).append({"role": "assistant", "content": assistant_msg})

    manager.load = AsyncMock(side_effect=load)
    manager.save_turn = AsyncMock(side_effect=save_turn)
    manager.delete_session = AsyncMock(return_value=True)

    redis_store = AsyncMock()
    redis_store.ping = AsyncMock(return_value=True)
    postgres_store = AsyncMock()
    postgres_store.ping = AsyncMock(return_value=True)
    manager.redis = redis_store
    manager.postgres = postgres_store
    return manager


@pytest.fixture
def benchmark_client(mock_retriever, mock_llm, mock_weather_adapter, mock_memory_manager):
    runtime_deps = RuntimeDeps(
        llm=mock_llm,
        retriever=mock_retriever,
        weather_adapter=mock_weather_adapter,
        memory_manager=mock_memory_manager,
    )
    app = create_app(
        memory_manager=mock_memory_manager,
        runtime_deps=runtime_deps,
        skip_db_init=True,
        skip_auto_ingest=True,
    )
    with TestClient(app) as client:
        yield client


def _make_chat_fn(client: TestClient):
    def chat(session_id: str | None, message: str) -> dict:
        payload = {"message": message}
        if session_id:
            payload["session_id"] = session_id
        started = time.perf_counter()
        response = client.post("/api/v1/chat", json=payload)
        latency_ms = (time.perf_counter() - started) * 1000
        result = {"status_code": response.status_code, "latency_ms": latency_ms}
        if response.status_code == 200:
            result.update(response.json())
        else:
            result["error"] = response.text
        return result

    return chat


@pytest.mark.benchmark
def test_benchmark_runner_executes_all_tasks(benchmark_client):
    runner = BenchmarkRunner(chat_fn=_make_chat_fn(benchmark_client))
    results = runner.run_all()

    assert len(results) >= 5
    assert all(result.task_id for result in results)
    assert all(result.turns for result in results)


@pytest.mark.benchmark
def test_benchmark_metrics_and_report(benchmark_client, tmp_path):
    runner = BenchmarkRunner(chat_fn=_make_chat_fn(benchmark_client))
    results = runner.run_all()
    report = write_report(results, tmp_path / "agent_benchmark_report.json")

    assert report.task_count == len(results)
    assert 0.0 <= report.summary["task_success_rate"] <= 1.0
    assert "intent_accuracy" in report.summary
    assert "overall_score" in report.summary
    assert report.by_intent

    report_path = tmp_path / "agent_benchmark_report.json"
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["task_count"] == len(results)
    assert "summary" in payload
    assert "by_intent" in payload


@pytest.mark.benchmark
def test_benchmark_report_includes_diff(benchmark_client, tmp_path):
    runner = BenchmarkRunner(chat_fn=_make_chat_fn(benchmark_client))
    results = runner.run_all()
    report_path = tmp_path / "agent_benchmark_report.json"

    write_report(results, report_path)
    second_report = write_report(results, report_path)

    assert second_report.diff_from_previous
    assert second_report.diff_from_previous.get("task_success_rate") == 0.0


@pytest.mark.benchmark
def test_single_turn_legal_task_passes(benchmark_client):
    runner = BenchmarkRunner(chat_fn=_make_chat_fn(benchmark_client))
    tasks = {task["id"]: task for task in runner.load_tasks()}
    result = runner.run_task(tasks["single_turn_legal"])

    assert result.task_success
    assert result.turns[0].intent == "legal"
    assert result.turns[0].citations
    assert result.turns[0].disclaimer


@pytest.mark.benchmark
def test_compute_metrics_structure(benchmark_client):
    runner = BenchmarkRunner(chat_fn=_make_chat_fn(benchmark_client))
    results = runner.run_all()
    report = compute_metrics(results)

    assert report.summary["latency_p95_ms"] >= 0
    assert "tool_success_rate" in report.summary
    assert "citation_compliance_rate" in report.summary
