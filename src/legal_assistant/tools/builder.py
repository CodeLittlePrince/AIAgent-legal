"""组装 Agent 可用的全部 StructuredTool。"""

from __future__ import annotations

from langchain_core.tools import StructuredTool

from legal_assistant.runtime.deps import RuntimeDeps
from legal_assistant.tools.context import AgentToolContext
from legal_assistant.tools.legal.search import create_legal_search_tool
from legal_assistant.tools.weather.forecast import create_weather_forecast_tool


def build_agent_tools(deps: RuntimeDeps, ctx: AgentToolContext) -> list[StructuredTool]:
    """注册并返回 Agent Tool Calling 使用的工具列表。"""
    return [
        create_legal_search_tool(deps, ctx),
        create_weather_forecast_tool(deps, ctx),
    ]
