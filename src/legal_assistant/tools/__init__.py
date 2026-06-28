"""工具层：外部能力适配器 + Agent StructuredTool。

具体 tool 请从子模块导入，例如 ``legal_assistant.tools.builder.build_agent_tools``。
"""

from legal_assistant.tools.constants import LEGAL_SEARCH_TOOL, WEATHER_FORECAST_TOOL

__all__ = ["LEGAL_SEARCH_TOOL", "WEATHER_FORECAST_TOOL"]
