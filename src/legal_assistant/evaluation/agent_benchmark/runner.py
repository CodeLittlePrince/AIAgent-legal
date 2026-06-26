from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

DEFAULT_TASKS_PATH = Path(__file__).resolve().parent / "tasks.yaml"


@dataclass
class TurnResult:
    turn_index: int
    user_message: str
    status_code: int
    intent: str | None = None
    answer: str | None = None
    citations: list[dict[str, str]] = field(default_factory=list)
    disclaimer: str | None = None
    trace_id: str | None = None
    latency_ms: float = 0.0
    error: str | None = None
    expect: dict[str, Any] = field(default_factory=dict)
    checks: dict[str, bool] = field(default_factory=dict)


@dataclass
class TaskRunResult:
    task_id: str
    task_name: str
    category: str
    session_id: str
    turns: list[TurnResult] = field(default_factory=list)
    task_success: bool = False


ChatFn = Callable[[str | None, str], dict[str, Any]]


class BenchmarkRunner:
    """Execute benchmark tasks against the /api/v1/chat endpoint."""

    def __init__(
        self,
        *,
        tasks_path: Path | str | None = None,
        base_url: str = "http://localhost:8000",
        chat_fn: ChatFn | None = None,
        http_client: httpx.Client | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.tasks_path = Path(tasks_path) if tasks_path else DEFAULT_TASKS_PATH
        self.base_url = base_url.rstrip("/")
        self.chat_fn = chat_fn
        self._http_client = http_client
        self.timeout = timeout
        self._owns_client = False

    def load_tasks(self) -> list[dict[str, Any]]:
        with self.tasks_path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return list(data.get("tasks") or [])

    def _get_client(self) -> httpx.Client:
        if self._http_client is None:
            self._http_client = httpx.Client(base_url=self.base_url, timeout=self.timeout)
            self._owns_client = True
        return self._http_client

    def close(self) -> None:
        if self._owns_client and self._http_client is not None:
            self._http_client.close()
            self._http_client = None
            self._owns_client = False

    def __enter__(self) -> BenchmarkRunner:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _default_chat(self, session_id: str | None, message: str) -> dict[str, Any]:
        client = self._get_client()
        payload: dict[str, str] = {"message": message}
        if session_id:
            payload["session_id"] = session_id

        started = time.perf_counter()
        response = client.post("/api/v1/chat", json=payload)
        latency_ms = (time.perf_counter() - started) * 1000

        result: dict[str, Any] = {
            "status_code": response.status_code,
            "latency_ms": latency_ms,
        }
        if response.status_code == 200:
            result.update(response.json())
        else:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            result["error"] = str(detail)
        return result

    def run_task(self, task: dict[str, Any]) -> TaskRunResult:
        chat = self.chat_fn or self._default_chat
        session_id = str(uuid.uuid4())
        turns_spec = task.get("turns") or []
        turn_results: list[TurnResult] = []

        for index, turn_spec in enumerate(turns_spec):
            user_message = turn_spec["user"]
            expect = dict(turn_spec.get("expect") or {})
            raw = chat(session_id, user_message)

            if raw.get("session_id"):
                session_id = raw["session_id"]

            turn_result = TurnResult(
                turn_index=index,
                user_message=user_message,
                status_code=int(raw.get("status_code", 500)),
                intent=raw.get("intent"),
                answer=raw.get("answer"),
                citations=list(raw.get("citations") or []),
                disclaimer=raw.get("disclaimer"),
                trace_id=raw.get("trace_id"),
                latency_ms=float(raw.get("latency_ms", 0.0)),
                error=raw.get("error"),
                expect=expect,
            )
            turn_results.append(turn_result)

        task_success = all(
            turn.status_code == 200 and _evaluate_turn(turn, turn_results)
            for turn in turn_results
        )
        return TaskRunResult(
            task_id=task["id"],
            task_name=task.get("name", task["id"]),
            category=task.get("category", "unknown"),
            session_id=session_id,
            turns=turn_results,
            task_success=task_success,
        )

    def run_all(self) -> list[TaskRunResult]:
        return [self.run_task(task) for task in self.load_tasks()]


def _evaluate_turn(turn: TurnResult, prior_turns: list[TurnResult]) -> bool:
    expect = turn.expect
    if not expect:
        return turn.status_code == 200

    checks: dict[str, bool] = {}

    if "intent" in expect:
        checks["intent"] = turn.intent == expect["intent"]

    if expect.get("citations") is True:
        checks["citations"] = len(turn.citations) > 0
    elif expect.get("citations") is False:
        checks["citations"] = len(turn.citations) == 0

    if expect.get("disclaimer") is True:
        checks["disclaimer"] = turn.disclaimer is not None and bool(turn.disclaimer.strip())
    elif expect.get("disclaimer") is False:
        checks["disclaimer"] = turn.disclaimer is None

    if expect.get("tool_success") is True:
        checks["tool_success"] = (
            turn.status_code == 200
            and turn.intent == "weather"
            and bool(turn.answer)
        )

    keywords = expect.get("answer_contains_any") or []
    if keywords:
        answer = (turn.answer or "").lower()
        checks["answer_contains_any"] = any(kw.lower() in answer for kw in keywords)

    if expect.get("memory_coherence") is True:
        memory_keywords = expect.get("memory_keywords_any") or []
        context_turn_index = expect.get("memory_context_from_turn")
        context_text = ""
        if context_turn_index is not None and 0 <= context_turn_index < len(prior_turns):
            context_turn = prior_turns[context_turn_index]
            context_text = context_turn.user_message or ""
        answer = turn.answer or ""
        combined = f"{context_text} {answer}"
        checks["memory_coherence"] = any(kw in combined for kw in memory_keywords)

    if expect.get("task_success") is True:
        checks["task_success"] = all(checks.values()) if checks else turn.status_code == 200

    turn.checks = checks
    return all(checks.values()) if checks else turn.status_code == 200
