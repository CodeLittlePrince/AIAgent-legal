"""Agent 基准测试指标汇总与 JSON 报告生成。

从 ``TaskRunResult`` 列表中聚合任务成功率、意图准确率、工具成功率、
记忆连贯率、引用合规率、延迟 P95 等，并可选择性地与上一份报告对比 diff。

主要入口：
- ``compute_metrics``：纯计算，返回 ``BenchmarkReport``
- ``write_report``：计算 + 写文件 + 与历史报告 diff
"""

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
    """一次 benchmark 运行的完整报告结构。

    Attributes:
        timestamp: ISO 8601  UTC 时间戳，报告生成时刻。
        task_count: 参与统计的任务数量。
        summary: 全局汇总指标（成功率、各维度比率、P95 延迟、综合分等）。
        by_intent: 按意图（如 legal、weather）分组的子指标。
        tasks: 各任务的序列化明细（含每轮 checks），便于人工复查。
        diff_from_previous: 与上一份同路径报告相比，summary 各数值型字段的差值；
            首次运行或无历史文件时为 empty dict。
    """

    timestamp: str
    task_count: int
    summary: dict[str, float | int]
    by_intent: dict[str, dict[str, float | int]]
    tasks: list[dict[str, Any]]
    diff_from_previous: dict[str, float | None] = field(default_factory=dict)


def _turn_intent_success(turn: TurnResult) -> bool | None:
    """若该轮 expect 中声明了 intent，则返回是否匹配；否则不参与 intent 统计。"""
    if "intent" not in turn.expect:
        return None
    return turn.intent == turn.expect["intent"]


def _turn_tool_success(turn: TurnResult) -> bool | None:
    """若 expect 要求 tool_success，则返回 checks 中的 tool_success；否则 None。"""
    if not turn.expect.get("tool_success"):
        return None
    return turn.checks.get("tool_success", False)


def _turn_memory_coherence(turn: TurnResult) -> bool | None:
    """若 expect 要求 memory_coherence，则返回对应 check；否则 None。"""
    if not turn.expect.get("memory_coherence"):
        return None
    return turn.checks.get("memory_coherence", False)


def _turn_citation_compliance(turn: TurnResult) -> bool | None:
    """判断该轮是否满足引用/免责声明合规（用于 legal 等场景）。

    规则概要：
    - 非 legal 且未强制要求 citations 时，不参与 citation 统计（返回 None）
    - legal 意图或 expect.citations=True 时：需有 citations
    - 若 expect.disclaimer 不为 False，还需非空 disclaimer

    Returns:
        True/False 表示是否合规；None 表示该轮不计入 citation_compliance_rate。
    """
    if turn.intent != "legal" and turn.expect.get("citations") is not True:
        return None
    if turn.expect.get("citations") is True or turn.intent == "legal":
        has_citations = len(turn.citations) > 0
        has_disclaimer = turn.disclaimer is not None and bool(turn.disclaimer.strip())
        if turn.expect.get("disclaimer") is False:
            return has_citations
        return has_citations and has_disclaimer
    return None


def _rate(values: list[bool]) -> float:
    """计算布尔列表中 True 的比例；空列表时返回 1.0（无样本视为满分，避免拉低 overall）。"""
    if not values:
        return 1.0
    return sum(1 for value in values if value) / len(values)


def compute_metrics(results: list[TaskRunResult]) -> BenchmarkReport:
    """从任务运行结果计算汇总指标并构建 ``BenchmarkReport``。

    遍历所有任务与轮次，收集 intent/tool/memory/citation 校验结果及延迟，
    并按首轮 intent 分组统计 task_success；最后计算 overall_score 为五项比率均值。

    Args:
        results: ``BenchmarkRunner.run_all()`` 或等价方式产生的列表。

    Returns:
        未写入磁盘的 ``BenchmarkReport``（``diff_from_previous`` 为空，由 write_report 填充）。
    """
    task_success_flags = [result.task_success for result in results]

    intent_checks: list[bool] = []
    tool_checks: list[bool] = []
    memory_checks: list[bool] = []
    citation_checks: list[bool] = []
    latencies: list[float] = []

    by_intent: dict[str, dict[str, list[bool | float]]] = defaultdict(
        lambda: {
            "task_success": [],
            "intent_accuracy": [],
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
                        "intent": turn.intent,
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

            intent_ok = _turn_intent_success(turn)
            if intent_ok is not None:
                intent_checks.append(intent_ok)
                if turn.intent:
                    by_intent[turn.intent]["intent_accuracy"].append(intent_ok)
                    by_intent[turn.intent]["count"].append(1.0)

            tool_ok = _turn_tool_success(turn)
            if tool_ok is not None:
                tool_checks.append(tool_ok)

            memory_ok = _turn_memory_coherence(turn)
            if memory_ok is not None:
                memory_checks.append(memory_ok)

            citation_ok = _turn_citation_compliance(turn)
            if citation_ok is not None:
                citation_checks.append(citation_ok)

        # 用任务第一轮 intent 作为该任务在 by_intent 分组的主标签
        primary_intent = result.turns[0].intent if result.turns else "unknown"
        if primary_intent:
            by_intent[primary_intent]["task_success"].append(result.task_success)

    latency_p95 = _percentile(latencies, 95) if latencies else 0.0

    summary = {
        "task_success_rate": _rate(task_success_flags),
        "intent_accuracy": _rate(intent_checks),
        "tool_success_rate": _rate(tool_checks),
        "memory_coherence_rate": _rate(memory_checks),
        "citation_compliance_rate": _rate(citation_checks),
        "latency_p95_ms": round(latency_p95, 2),
        "cost_per_task_usd": 0.0,  # 占位字段，后续可接入 token 成本统计
        "overall_score": 0.0,
    }

    # 综合分：五项比率（不含延迟与成本）的算术平均
    summary["overall_score"] = round(
        (
            summary["task_success_rate"]
            + summary["intent_accuracy"]
            + summary["tool_success_rate"]
            + summary["memory_coherence_rate"]
            + summary["citation_compliance_rate"]
        )
        / 5,
        4,
    )

    by_intent_summary: dict[str, dict[str, float | int]] = {}
    for intent, values in by_intent.items():
        task_success_list = [bool(v) for v in values["task_success"]]
        intent_accuracy_list = [bool(v) for v in values["intent_accuracy"]]
        by_intent_summary[intent] = {
            "task_success_rate": round(_rate(task_success_list), 4),
            "intent_accuracy": round(_rate(intent_accuracy_list), 4),
            "count": len(task_success_list) or len(intent_accuracy_list),
        }

    return BenchmarkReport(
        timestamp=datetime.now(UTC).isoformat(),
        task_count=len(results),
        summary=summary,
        by_intent=by_intent_summary,
        tasks=serialized_tasks,
    )


def _percentile(values: list[float], percentile: float) -> float:
    """计算数值列表的近似百分位数（线性索引法，非插值）。

    Args:
        values: 非空浮点列表（调用方保证或返回 0.0）。
        percentile: 0–100，如 95 表示 P95。

    Returns:
        排序后对应索引位置的值。
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((percentile / 100) * (len(ordered) - 1)))))
    return ordered[index]


def _load_previous_report(path: Path) -> dict[str, Any] | None:
    """若输出路径已存在，读取上一份 JSON 报告；否则返回 None。"""
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _compute_diff(
    current: dict[str, float | int],
    previous: dict[str, Any] | None,
) -> dict[str, float | None]:
    """计算当前 summary 与上一份报告 summary 的数值差（current - previous）。

    仅处理 int/float 字段；上一份缺少对应键时 diff 值为 None。
    """
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
    """计算指标、写入 JSON 文件，并填充与上一份报告的 diff。

    若 ``output_path`` 已存在，会先读取旧报告再计算 ``diff_from_previous``，
    便于对比两次 benchmark 运行的指标变化。

    Args:
        results: 任务运行结果列表。
        output_path: 输出 JSON 路径；父目录不存在时会自动创建。

    Returns:
        写入前的完整 ``BenchmarkReport``（含 diff 字段）。
    """
    path = Path(output_path)
    report = compute_metrics(results)
    previous = _load_previous_report(path)
    report.diff_from_previous = _compute_diff(report.summary, previous)

    payload = asdict(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return report
