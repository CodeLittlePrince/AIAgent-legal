"""天气查询 Tool（封装 ``WeatherAdapter`` 供 Agent Tool Calling 使用）。"""

from __future__ import annotations

from langchain_core.tools import StructuredTool

from legal_assistant.runtime.deps import RuntimeDeps
from legal_assistant.tools.constants import WEATHER_FORECAST_TOOL
from legal_assistant.tools.context import AgentToolContext


def create_weather_forecast_tool(deps: RuntimeDeps, ctx: AgentToolContext) -> StructuredTool:
    """构建 ``get_weather_forecast`` StructuredTool。"""

    async def get_weather_forecast(location: str) -> str:
        """查询指定城市的当前天气与简要预报。location 为中文城市名。"""
        ctx.tools_used.add(WEATHER_FORECAST_TOOL)
        weather = await deps.get_weather_adapter().get_weather(location)
        ctx.weather_result = weather
        return (
            f"地点: {weather.location}\n"
            f"温度: {weather.temperature}\n"
            f"天气: {weather.conditions}\n"
            f"预报: {weather.forecast_summary}"
        )

    return StructuredTool.from_function(
        coroutine=get_weather_forecast,
        name=WEATHER_FORECAST_TOOL,
        description="查询中国城市天气与气温预报。",
    )
