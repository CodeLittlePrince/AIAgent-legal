"""工具层公共接口。

本包封装外部能力（如天气查询）的适配器与注册逻辑，
上层 runtime 节点通过此处导出的类型与工厂函数获取具体实现。
"""

from legal_assistant.tools.base import WeatherAdapter, WeatherResult
from legal_assistant.tools.registry import get_weather_adapter

__all__ = ["WeatherAdapter", "WeatherResult", "get_weather_adapter"]
