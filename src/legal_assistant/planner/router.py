from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from legal_assistant.planner.intent import analyze_by_rules

LLMClassifier = Callable[[str, list[dict[str, Any]]], Awaitable["PlanResult"]]

HIGH_CONFIDENCE_THRESHOLD = 0.85


@dataclass
class PlanResult:
    intent: str
    confidence: float
    location: str | None = None


async def classify(
    message: str,
    history: list[dict[str, Any]] | None = None,
    llm_classifier: LLMClassifier | None = None,
) -> PlanResult:
    history = history or []
    intent, confidence, location = analyze_by_rules(message)

    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return PlanResult(intent=intent, confidence=confidence, location=location)

    if llm_classifier is not None:
        return await llm_classifier(message, history)

    return PlanResult(intent=intent, confidence=confidence, location=location)
