from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from legal_assistant.config import settings
from legal_assistant.knowledge.legal_qa import (
    append_disclaimer,
    build_legal_prompt,
    format_citations,
)
from legal_assistant.knowledge.retriever import LegalRetriever
from legal_assistant.memory.manager import MemoryManager
from legal_assistant.planner.router import LLMClassifier, classify
from legal_assistant.runtime.state import AgentState
from legal_assistant.tools.base import WeatherAdapter
from legal_assistant.tools.registry import get_weather_adapter


def _last_user_message(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
    return ""


def _history_to_langchain(messages: list[dict[str, Any]]) -> list[Any]:
    converted: list[Any] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "user":
            converted.append(HumanMessage(content=content))
        elif role == "assistant":
            converted.append(AIMessage(content=content))
        elif role == "system":
            converted.append(SystemMessage(content=content))
    return converted


def create_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
    )


@dataclass
class RuntimeDeps:
    llm: BaseChatModel | None = None
    retriever: LegalRetriever | None = None
    memory_manager: MemoryManager | None = None
    weather_adapter: WeatherAdapter | None = None
    llm_classifier: LLMClassifier | None = None

    def get_llm(self) -> BaseChatModel:
        return self.llm or create_llm()

    def get_retriever(self) -> LegalRetriever:
        return self.retriever or LegalRetriever()

    def get_memory_manager(self) -> MemoryManager:
        return self.memory_manager or MemoryManager()

    def get_weather_adapter(self) -> WeatherAdapter:
        return self.weather_adapter or get_weather_adapter(settings.weather_provider)


def make_planner_node(deps: RuntimeDeps) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    async def planner_node(state: AgentState) -> dict[str, Any]:
        message = _last_user_message(state.get("messages", []))
        history = state.get("messages", [])
        try:
            result = await classify(message, history, llm_classifier=deps.llm_classifier)
            return {
                "intent": result.intent,
                "location": result.location,
                "error": None,
            }
        except Exception as exc:
            return {"intent": "general", "error": str(exc)}

    return planner_node


def make_legal_node(deps: RuntimeDeps) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    async def legal_node(state: AgentState) -> dict[str, Any]:
        message = _last_user_message(state.get("messages", []))
        try:
            retriever = deps.get_retriever()
            docs = retriever.retrieve(message)
            prompt = build_legal_prompt(message, docs)
            response = await deps.get_llm().ainvoke([HumanMessage(content=prompt)])
            answer = append_disclaimer(str(response.content))
            citations = format_citations(docs) if docs else []
            return {
                "retrieved_docs": docs,
                "answer": answer,
                "citations": citations,
                "error": None,
            }
        except Exception as exc:
            return {"error": str(exc), "answer": None, "citations": []}

    return legal_node


def make_weather_node(deps: RuntimeDeps) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    async def weather_node(state: AgentState) -> dict[str, Any]:
        message = _last_user_message(state.get("messages", []))
        location = state.get("location") or message
        try:
            weather = await deps.get_weather_adapter().get_weather(location)
            prompt = (
                "你是天气助手。请根据以下天气数据，用简洁中文回答用户。\n"
                f"地点: {weather.location}\n"
                f"温度: {weather.temperature}\n"
                f"天气: {weather.conditions}\n"
                f"预报: {weather.forecast_summary}\n"
                f"用户问题: {message}"
            )
            response = await deps.get_llm().ainvoke([HumanMessage(content=prompt)])
            return {
                "tool_result": weather,
                "answer": str(response.content),
                "error": None,
            }
        except Exception as exc:
            return {"error": str(exc), "answer": None}

    return weather_node


def make_general_node(deps: RuntimeDeps) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    async def general_node(state: AgentState) -> dict[str, Any]:
        try:
            history = _history_to_langchain(state.get("messages", []))
            response = await deps.get_llm().ainvoke(history)
            return {"answer": str(response.content), "error": None}
        except Exception as exc:
            return {"error": str(exc), "answer": None}

    return general_node


def make_save_memory_node(
    deps: RuntimeDeps,
) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    async def save_memory_node(state: AgentState) -> dict[str, Any]:
        session_id = state.get("session_id")
        user_message = _last_user_message(state.get("messages", []))
        answer = state.get("answer")
        intent = state.get("intent")

        if not session_id or not user_message or not answer:
            return {}

        try:
            await deps.get_memory_manager().save_turn(
                session_id=session_id,
                user_msg=user_message,
                assistant_msg=answer,
                intent=intent,
            )
        except Exception as exc:
            return {"error": str(exc)}
        return {}

    return save_memory_node
