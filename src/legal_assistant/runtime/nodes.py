"""Agent 图节点：天气快速路径、Tool Calling Agent、记忆持久化。"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import RunnableConfig

from legal_assistant.observability.langchain_tracing import langchain_invoke_config
from legal_assistant.observability.tracing import span
from legal_assistant.planner.weather_rules import detect_weather_route
from legal_assistant.runtime.agent_loop import run_tool_agent
from legal_assistant.runtime.deps import RuntimeDeps
from legal_assistant.runtime.state import AgentNode, AgentState

__all__ = [
    "RuntimeDeps",
    "make_route_node",
    "make_weather_node",
    "make_agent_node",
    "make_save_memory_node",
]

from legal_assistant.tools.constants import LEGAL_SEARCH_TOOL, WEATHER_FORECAST_TOOL


def _last_user_message(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
    return ""


def make_route_node(deps: RuntimeDeps) -> AgentNode:
    """规则识别天气意图；其余交给 Tool Calling Agent。"""

    async def route_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        del config
        message = _last_user_message(state.get("messages", []))
        history = state.get("messages", [])
        try:
            with span("graph.route"):
                weather = detect_weather_route(message, history)
            if weather is not None:
                return {
                    "route": "weather",
                    "intent": "weather",
                    "location": weather.location,
                    "error": None,
                }
            return {"route": "agent", "intent": None, "location": None, "error": None}
        except Exception as exc:
            return {"route": "agent", "intent": None, "error": str(exc)}

    return route_node


def make_weather_node(deps: RuntimeDeps) -> AgentNode:
    """天气快速路径：规则已识别 → 直调天气 API → 单次 LLM 组织回复。"""

    async def weather_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        message = _last_user_message(state.get("messages", []))
        location = state.get("location") or message
        try:
            with span("tool.weather"):
                weather = await deps.get_weather_adapter().get_weather(location)
            prompt = (
                "你是天气助手。请根据以下天气数据，用简洁中文回答用户。\n"
                f"地点: {weather.location}\n"
                f"温度: {weather.temperature}\n"
                f"天气: {weather.conditions}\n"
                f"预报: {weather.forecast_summary}\n"
                f"用户问题: {message}"
            )
            response = await deps.get_llm().ainvoke(
                [HumanMessage(content=prompt)],
                config=langchain_invoke_config(),
            )
            return {
                "intent": "weather",
                "tools_used": [WEATHER_FORECAST_TOOL],
                "tool_result": weather,
                "location": weather.location,
                "answer": str(response.content),
                "citations": None,
                "error": None,
            }
        except Exception as exc:
            return {"error": str(exc), "answer": None}

    return weather_node


def make_agent_node(deps: RuntimeDeps) -> AgentNode:
    async def agent_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        try:
            result = await run_tool_agent(
                deps=deps,
                llm=deps.get_llm(),
                messages=state.get("messages", []),
                llm_config=langchain_invoke_config(),
            )
            if result.get("tools_used"):
                if LEGAL_SEARCH_TOOL in result["tools_used"]:
                    result["intent"] = "legal"
                elif WEATHER_FORECAST_TOOL in result["tools_used"]:
                    result["intent"] = "weather"
                else:
                    result["intent"] = "general"
            else:
                result["intent"] = "general"
            return result
        except Exception as exc:
            return {"error": str(exc), "answer": None}

    return agent_node


def make_save_memory_node(deps: RuntimeDeps) -> AgentNode:
    async def save_memory_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        del config
        session_id = state.get("session_id")
        user_message = _last_user_message(state.get("messages", []))
        answer = state.get("answer")

        if not session_id or not user_message or not answer:
            return {}

        try:
            with span("graph.save_memory"):
                await deps.get_memory_manager().save_turn(
                    session_id=session_id,
                    user_msg=user_message,
                    assistant_msg=answer,
                )
        except Exception as exc:
            return {"error": str(exc)}
        return {}

    return save_memory_node
