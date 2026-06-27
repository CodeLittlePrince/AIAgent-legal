"""意图路由模块。

组合规则分类与可选的 LLM 分类器：
规则置信度高时直接返回；否则可调用 LLM 做二次判断。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from legal_assistant.planner.intent import analyze_by_rules

# LLM 分类器的类型别名：接收 (消息, 历史) 异步返回 PlanResult
LLMClassifier = Callable[[str, list[dict[str, Any]]], Awaitable["PlanResult"]]

# 达到此置信度阈值时，规则结果足够可靠，无需再调 LLM
HIGH_CONFIDENCE_THRESHOLD = 0.85


@dataclass
class PlanResult:
    """Planner 的输出：意图规划结果。

    Attributes:
        intent: 最终意图（legal / weather / general）。
        confidence: 置信度 0~1，越高表示越确定。
        location: 天气查询时的地点；其他意图通常为 None。
    """

    intent: str
    confidence: float
    location: str | None = None


async def classify(
    message: str,
    history: list[dict[str, Any]] | None = None,
    llm_classifier: LLMClassifier | None = None,
) -> PlanResult:
    """对用户消息进行意图分类（规则优先，低置信度时可走 LLM）。

    流程：
    1. 先用 analyze_by_rules 快速分析；
    2. 若 confidence >= 0.85，直接返回规则结果；
    3. 否则若提供了 llm_classifier，交给 LLM 判断；
    4. 仍无 LLM 时，返回规则结果（可能置信度较低）。

    Args:
        message: 当前用户消息。
        history: 会话历史，供 LLM 分类器参考上下文。
        llm_classifier: 可选的异步 LLM 分类函数。

    Returns:
        包含 intent、confidence、location 的 PlanResult。
    """
    history = history or []
    intent, confidence, location = analyze_by_rules(message)

    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return PlanResult(intent=intent, confidence=confidence, location=location)

    if llm_classifier is not None:
        return await llm_classifier(message, history)

    return PlanResult(intent=intent, confidence=confidence, location=location)
