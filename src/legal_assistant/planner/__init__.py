"""Planner：仅保留天气规则快速路径，法律/闲聊走 Tool Calling Agent。"""

from legal_assistant.planner.weather_rules import WeatherRoute, detect_weather_route

__all__ = ["WeatherRoute", "detect_weather_route"]
