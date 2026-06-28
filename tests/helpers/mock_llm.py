"""Shared LLM mocks compatible with LangChain ``create_agent``."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from pydantic import ConfigDict


def _last_user_message(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if message.__class__.__name__ == "HumanMessage":
            return str(getattr(message, "content", ""))
    return ""


def _has_tool_message(messages: list[BaseMessage]) -> bool:
    return any(message.__class__.__name__ == "ToolMessage" for message in messages)


def _is_agent_invocation(messages: list[BaseMessage]) -> bool:
    for message in messages:
        if message.__class__.__name__ == "SystemMessage":
            content = str(getattr(message, "content", ""))
            if "search_legal_knowledge" in content:
                return True
    return False


def _build_agent_response(messages: list[BaseMessage]) -> AIMessage:
    user_msg = _last_user_message(messages)

    if _is_agent_invocation(messages) and not _has_tool_message(messages):
        if any(keyword in user_msg for keyword in ("试用期", "劳动", "法律", "工资", "合同", "辞退", "工伤")):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_legal_knowledge",
                        "args": {"query": user_msg},
                        "id": "call_legal_1",
                    }
                ],
            )
        if any(keyword in user_msg for keyword in ("天气", "气温", "下雨", "预报", "明天")):
            location = "北京"
            for city in ("北京", "上海", "广州", "深圳", "杭州"):
                if city in user_msg:
                    location = city
                    break
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_weather_forecast",
                        "args": {"location": location},
                        "id": "call_weather_1",
                    }
                ],
            )

    if any(keyword in user_msg for keyword in ("试用期", "劳动", "法律", "合同")):
        return AIMessage(content="试用期最长不超过六个月。")
    if any(keyword in user_msg for keyword in ("天气", "气温", "明天", "预报")):
        location = "上海" if "上海" in user_msg else "北京"
        if "明天" in user_msg:
            return AIMessage(content=f"{location}明天多云，气温18到25度，适合出行。")
        return AIMessage(content=f"{location}今天晴，气温22度，未来三天以晴为主。")
    return AIMessage(content="你好，我是智能法律助手，可以解答法律、天气和一般问题。")


class MockLegalChatModel(BaseChatModel):
    """Minimal chat model for tests using ``create_agent``."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bound_tools: list[Any] | None = None

    @property
    def _llm_type(self) -> str:
        return "mock-legal-chat"

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> Runnable:
        self.bound_tools = list(tools)
        return self

    def _resolve_response(self, messages: list[BaseMessage]) -> AIMessage:
        for message in messages:
            content = str(getattr(message, "content", ""))
            if "你是天气助手" in content or ("地点:" in content and "温度:" in content):
                location = "北京"
                for city in ("北京", "上海", "广州", "深圳", "杭州"):
                    if city in content:
                        location = city
                        break
                return AIMessage(content=f"{location}今天晴，22度，未来三天以晴为主。")
        return _build_agent_response(messages)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        generation = ChatGeneration(message=self._resolve_response(messages))
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def make_tool_calling_mock_llm() -> MockLegalChatModel:
    """Mock LLM：法律/天气先 tool_call，闲聊直接文本回复。"""
    return MockLegalChatModel()
