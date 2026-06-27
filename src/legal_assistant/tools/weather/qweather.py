"""和风天气（QWeather）适配器占位实现。

当前尚未对接和风 API；调用 ``get_weather`` 会抛出 ``NotImplementedError``。
后续实现时需：地理编码 → 实时天气 → 预报，并映射为 ``WeatherResult``。
"""

from legal_assistant.tools.base import WeatherResult


class QWeatherAdapter:
    """和风天气 API 适配器（待实现）。"""

    async def get_weather(self, location: str) -> WeatherResult:
        """查询指定地点天气（尚未实现）。

        Args:
            location: 用户输入或解析出的地点名称。

        Raises:
            NotImplementedError: 当前版本未实现和风接口。
        """
        raise NotImplementedError("QWeatherAdapter is not implemented yet")
