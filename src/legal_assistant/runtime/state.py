"""Agent 运行时状态定义。"""

from collections.abc import Awaitable
from typing import Any, Protocol, TypedDict

from legal_assistant.knowledge.retriever import RetrievedDoc
from legal_assistant.tools.base import WeatherResult


class AgentState(TypedDict, total=False):
    """Agent 图执行时的共享状态。"""

    session_id: str
    messages: list[dict[str, Any]]
    route: str | None
    intent: str | None
    tools_used: list[str] | None
    location: str | None
    retrieved_docs: list[RetrievedDoc] | None
    tool_result: WeatherResult | None
    answer: str | None
    citations: list[dict[str, str]] | None
    error: str | None


class AgentNode(Protocol):
    def __call__(self, state: AgentState) -> Awaitable[dict[str, Any]]: ...
