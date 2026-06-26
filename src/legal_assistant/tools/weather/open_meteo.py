from typing import Any

import httpx

from legal_assistant.tools.base import WeatherResult

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

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
    if code is None:
        return "未知"
    return WMO_CONDITIONS.get(code, f"天气代码 {code}")


def _format_location(result: dict[str, Any]) -> str:
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
    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client

    async def get_weather(self, location: str) -> WeatherResult:
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=10.0)
        try:
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
            if owns_client:
                await client.aclose()
