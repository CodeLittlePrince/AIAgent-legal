"""Agent 基准测试指标汇总与 JSON 报告生成。"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from legal_assistant.evaluation.agent_benchmark.runner import TaskRunResult, TurnResult


@dataclass
class BenchmarkReport:
    timestamp: str
    task_count: int
    summary: dict[str, float | int]
    by_tool: dict[str, dict[str, float | int]]
    tasks: list[dict[str, Any]]
    diff_from_previous: dict[str, float | None] = field(default_factory=dict)


def _turn_tool_selection_success(turn: TurnResult) -> bool | None:
    expect = turn.expect
    if isinstance(expect.get("tools_contains"), str):
        return expect["tools_contains"] in turn.tools_used
    if expect.get("tools_empty") is True:
        return len(turn.tools_used) == 0
    return None


def _turn_tool_success(turn: TurnResult) -> bool | None:
    if not turn.expect.get("tool_success"):
        return None
    return turn.checks.get("tool_success", False)


def _turn_memory_coherence(turn: TurnResult) -> bool | None:
    if not turn.expect.get("memory_coherence"):
        return None
    return turn.checks.get("memory_coherence", False)


def _turn_citation_compliance(turn: TurnResult) -> bool | None:
    if (
        "search_legal_knowledge" not in turn.tools_used
        and turn.expect.get("citations") is not True
    ):
        return None
    if turn.expect.get("citations") is True or "search_legal_knowledge" in turn.tools_used:
        has_citations = len(turn.citations) > 0
        has_disclaimer = turn.disclaimer is not None and bool(turn.disclaimer.strip())
        if turn.expect.get("disclaimer") is False:
            return has_citations
        return has_citations and has_disclaimer
    return None


def _primary_tool(tools_used: list[str]) -> str:
    if not tools_used:
        return "none"
    return tools_used[0]


def _rate(values: list[bool]) -> float:
    if not values:
        return 1.0
    return sum(1 for value in values if value) / len(values)


def compute_metrics(results: list[TaskRunResult]) -> BenchmarkReport:
    task_success_flags = [result.task_success for result in results]

    tool_selection_checks: list[bool] = []
    tool_checks: list[bool] = []
    memory_checks: list[bool] = []
    citation_checks: list[bool] = []
    latencies: list[float] = []

    by_tool: dict[str, dict[str, list[bool | float]]] = defaultdict(
        lambda: {
            "task_success": [],
            "tool_selection_accuracy": [],
            "count": [],
        }
    )

    serialized_tasks: list[dict[str, Any]] = []

    for result in results:
        serialized_tasks.append(
            {
                "task_id": result.task_id,
                "task_name": result.task_name,
                "category": result.category,
                "session_id": result.session_id,
                "task_success": result.task_success,
                "turns": [
                    {
                        "turn_index": turn.turn_index,
                        "user_message": turn.user_message,
                        "status_code": turn.status_code,
                        "tools_used": turn.tools_used,
                        "latency_ms": turn.latency_ms,
                        "checks": turn.checks,
                        "error": turn.error,
                    }
                    for turn in result.turns
                ],
            }
        )

        for turn in result.turns:
            latencies.append(turn.latency_ms)

            tool_selection_ok = _turn_tool_selection_success(turn)
            if tool_selection_ok is not None:
                tool_selection_checks.append(tool_selection_ok)
                primary = _primary_tool(turn.tools_used)
                by_tool[primary]["tool_selection_accuracy"].append(tool_selection_ok)
                by_tool[primary]["count"].append(1.0)

            tool_ok = _turn_tool_success(turn)
            if tool_ok is not None:
                tool_checks.append(tool_ok)

            memory_ok = _turn_memory_coherence(turn)
            if memory_ok is not None:
                memory_checks.append(memory_ok)

            citation_ok = _turn_citation_compliance(turn)
            if citation_ok is not None:
                citation_checks.append(citation_ok)

        primary_tool = _primary_tool(result.turns[0].tools_used if result.turns else [])
        by_tool[primary_tool]["task_success"].append(result.task_success)

    latency_p95 = _percentile(latencies, 95) if latencies else 0.0

    summary = {
        "task_success_rate": _rate(task_success_flags),
        "tool_selection_accuracy": _rate(tool_selection_checks),
        "tool_success_rate": _rate(tool_checks),
        "memory_coherence_rate": _rate(memory_checks),
        "citation_compliance_rate": _rate(citation_checks),
        "latency_p95_ms": round(latency_p95, 2),
        "cost_per_task_usd": 0.0,
        "overall_score": 0.0,
    }

    summary["overall_score"] = round(
        (
            summary["task_success_rate"]
            + summary["tool_selection_accuracy"]
            + summary["tool_success_rate"]
            + summary["memory_coherence_rate"]
            + summary["citation_compliance_rate"]
        )
        / 5,
        4,
    )

    by_tool_summary: dict[str, dict[str, float | int]] = {}
    for tool_name, values in by_tool.items():
        task_success_list = [bool(v) for v in values["task_success"]]
        selection_list = [bool(v) for v in values["tool_selection_accuracy"]]
        by_tool_summary[tool_name] = {
            "task_success_rate": round(_rate(task_success_list), 4),
            "tool_selection_accuracy": round(_rate(selection_list), 4),
            "count": len(task_success_list) or len(selection_list),
        }

    return BenchmarkReport(
        timestamp=datetime.now(UTC).isoformat(),
        task_count=len(results),
        summary=summary,
        by_tool=by_tool_summary,
        tasks=serialized_tasks,
    )


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((percentile / 100) * (len(ordered) - 1)))))
    return ordered[index]


def _load_previous_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _compute_diff(
    current: dict[str, float | int],
    previous: dict[str, Any] | None,
) -> dict[str, float | None]:
    if previous is None:
        return {}
    prev_summary = previous.get("summary") or {}
    diff: dict[str, float | None] = {}
    for key, value in current.items():
        if not isinstance(value, (int, float)):
            continue
        prev_value = prev_summary.get(key)
        if isinstance(prev_value, (int, float)):
            diff[key] = round(float(value) - float(prev_value), 4)
        else:
            diff[key] = None
    return diff


def write_report(
    results: list[TaskRunResult],
    output_path: Path | str = "agent_benchmark_report.json",
) -> BenchmarkReport:
    path = Path(output_path)
    report = compute_metrics(results)
    previous = _load_previous_report(path)
    report.diff_from_previous = _compute_diff(report.summary, previous)

    payload = asdict(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return report
