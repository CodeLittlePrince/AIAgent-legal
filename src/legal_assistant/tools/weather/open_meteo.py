"""Open-Meteo 免费天气 API 适配器。

Open-Meteo 无需 API Key：先通过地理编码 API 将地名转为经纬度，
再请求预报 API 获取当前天气与未来 3 日预报，并映射 WMO 天气代码为中文描述。
"""

from typing import Any

import httpx

from legal_assistant.tools.base import WeatherResult

# Open-Meteo 地理编码与预报接口的基础 URL
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO 标准天气现象代码 → 中文简述（Open-Meteo 返回的是整数 code）
WMO_CONDITIONS: dict[int, str] = {
    0: "晴",
    1: "大部晴朗",
    2: "局部多云",
    3: "多云",
    45: "雾",
    48: "雾凇",
    51: "小毛毛雨",
    53: "毛毛雨",
    55: "大毛毛雨",
    56: "冻毛毛雨",
    57: "冻毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "冻雨",
    67: "冻雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "雪粒",
    80: "小阵雨",
    81: "阵雨",
    82: "大阵雨",
    85: "小阵雪",
    86: "大阵雪",
    95: "雷暴",
    96: "雷暴伴小冰雹",
    99: "雷暴伴大冰雹",
}


def _weather_code_to_conditions(code: int | None) -> str:
    """将 WMO 天气代码转为中文状况描述。

    Args:
        code: Open-Meteo 返回的 ``weather_code``，可能为 ``None``。

    Returns:
        已知代码对应的中文；未知代码返回 ``"天气代码 {code}"``；无代码时返回「未知」。
    """
    if code is None:
        return "未知"
    return WMO_CONDITIONS.get(code, f"天气代码 {code}")


def _format_location(result: dict[str, Any]) -> str:
    """将地理编码结果格式化为可读地点字符串。

    组合地名、一级行政区（省/州）、国家，用逗号连接，例如「北京, 北京, 中国」。

    Args:
        result: 地理编码 API 返回的 ``results[0]`` 单条记录。

    Returns:
        逗号分隔的地点描述。
    """
    name = result.get("name", "")
    admin1 = result.get("admin1")
    country = result.get("country")
    parts = [name]
    if admin1:
        parts.append(admin1)
    if country:
        parts.append(country)
    return ", ".join(parts)


def _build_forecast_summary(forecast: dict[str, Any]) -> str:
    """从预报 JSON 的 daily 字段生成多日摘要文本。

    逐日输出：日期、天气状况、最低~最高温度（摄氏度，取整）。

    Args:
        forecast: 预报 API 的完整 JSON 响应。

    Returns:
        用中文分号连接的预报行；无 daily 数据时返回「暂无预报数据」。
    """
    daily = forecast.get("daily", {})
    dates = daily.get("time", [])
    max_temps = daily.get("temperature_2m_max", [])
    min_temps = daily.get("temperature_2m_min", [])
    codes = daily.get("weather_code", [])

    if not dates:
        return "暂无预报数据"

    lines: list[str] = []
    for i, date in enumerate(dates):
        code = codes[i] if i < len(codes) else None
        t_max = max_temps[i] if i < len(max_temps) else None
        t_min = min_temps[i] if i < len(min_temps) else None
        condition = _weather_code_to_conditions(code)
        if t_max is not None and t_min is not None:
            lines.append(f"{date}: {condition}，{t_min:.0f}°C ~ {t_max:.0f}°C")
        else:
            lines.append(f"{date}: {condition}")
    return "；".join(lines)


class OpenMeteoAdapter:
    """Open-Meteo 天气服务适配器，实现 ``WeatherAdapter`` 协议。"""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        """初始化适配器。

        Args:
            client: 可选的共享 ``httpx.AsyncClient``；为 ``None`` 时每次请求临时创建并在结束后关闭。
        """
        self._client = client

    async def get_weather(self, location: str) -> WeatherResult:
        """查询指定地点的当前天气与短期预报。

        步骤：
        1. 地理编码：地名 → 经纬度与规范地名
        2. 预报请求：当前温度/天气码 + 未来 3 日 daily 数据
        3. 组装为统一的 ``WeatherResult``

        Args:
            location: 用户输入的地点名称（支持中英文等地名）。

        Returns:
            标准化天气结果，``raw_source`` 中保留原始 geocoding 与 forecast JSON。

        Raises:
            ValueError: 地理编码未找到任何匹配地点。
            httpx.HTTPStatusError: HTTP 请求返回非 2xx 状态码。
        """
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=10.0, trust_env=False)
        try:
            # 第一步：地理编码
            geo_response = await client.get(
                GEOCODING_URL,
                params={"name": location, "count": 1, "language": "zh", "format": "json"},
            )
            geo_response.raise_for_status()
            geo_data = geo_response.json()
            results = geo_data.get("results") or []
            if not results:
                raise ValueError(f"未找到地点: {location}")

            place = results[0]
            latitude = place["latitude"]
            longitude = place["longitude"]
            resolved_location = _format_location(place)

            # 第二步：根据经纬度拉取预报（含 current 与 3 日 daily）
            forecast_response = await client.get(
                FORECAST_URL,
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "current": "temperature_2m,weather_code",
                    "daily": "weather_code,temperature_2m_max,temperature_2m_min",
                    "timezone": "auto",
                    "forecast_days": 3,
                },
            )
            forecast_response.raise_for_status()
            forecast_data = forecast_response.json()

            current = forecast_data.get("current", {})
            temperature = current.get("temperature_2m")
            conditions = _weather_code_to_conditions(current.get("weather_code"))

            return WeatherResult(
                location=resolved_location,
                temperature=temperature,
                conditions=conditions,
                forecast_summary=_build_forecast_summary(forecast_data),
                raw_source={"geocoding": geo_data, "forecast": forecast_data},
            )
        finally:
            # 仅在使用临时 client 时才关闭，避免关闭调用方注入的共享 client
            if owns_client:
                await client.aclose()
