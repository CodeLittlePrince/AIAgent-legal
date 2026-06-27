"""工具层的基础数据结构与协议定义。

通过 ``Protocol`` 定义天气适配器接口，不同供应商（Open-Meteo、和风、高德等）
只需实现同一套 ``get_weather`` 方法，即可被 registry 统一选用。
"""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class WeatherResult:
    """天气查询的标准化返回结构。

    各适配器将第三方 API 的原始 JSON 解析后，统一填充为本数据类，
    便于 LLM 节点用固定字段生成自然语言回答。

    Attributes:
        location: 解析后的地点名称（可能含省/国家等信息）。
        temperature: 当前气温（摄氏度），API 无数据时为 ``None``。
        conditions: 当前天气状况的中文描述，如「晴」「小雨」。
        forecast_summary: 未来若干天预报的简要文本摘要。
        raw_source: 原始 API 响应，便于调试或后续扩展字段。
    """

    location: str
    temperature: float | None
    conditions: str
    forecast_summary: str
    raw_source: dict[str, Any]


class WeatherAdapter(Protocol):
    """天气数据适配器协议（结构化子类型，无需显式继承）。

    任何实现了 ``async def get_weather(self, location: str) -> WeatherResult``
    的类都满足此协议，可在 registry 中注册使用。
    """

    async def get_weather(self, location: str) -> WeatherResult: ...
