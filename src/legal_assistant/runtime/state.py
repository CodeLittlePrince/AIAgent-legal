from typing import Any, TypedDict

from legal_assistant.knowledge.retriever import RetrievedDoc
from legal_assistant.tools.base import WeatherResult


class AgentState(TypedDict, total=False):
    session_id: str
    messages: list[dict[str, Any]]
    intent: str | None
    location: str | None
    retrieved_docs: list[RetrievedDoc] | None
    tool_result: WeatherResult | None
    answer: str | None
    citations: list[dict[str, str]] | None
    error: str | None
