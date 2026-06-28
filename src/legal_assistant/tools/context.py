"""Agent 单次执行过程中的 tool 副作用收集。"""

from __future__ import annotations

from dataclasses import dataclass, field

from legal_assistant.knowledge.retriever import RetrievedDoc
from legal_assistant.tools.base import WeatherResult


@dataclass
class AgentToolContext:
    retrieved_docs: list[RetrievedDoc] = field(default_factory=list)
    weather_result: WeatherResult | None = None
    tools_used: set[str] = field(default_factory=set)
