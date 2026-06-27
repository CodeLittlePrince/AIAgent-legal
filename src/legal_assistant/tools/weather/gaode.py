"""高德地图天气适配器占位实现。

当前尚未对接高德 Web 服务 API；调用 ``get_weather`` 会抛出 ``NotImplementedError``。
后续实现时需使用高德 Key，完成城市编码与天气实况/预报查询。
"""

from legal_assistant.tools.base import WeatherResult


class GaodeAdapter:
    """高德地图天气 API 适配器（待实现）。"""

    async def get_weather(self, location: str) -> WeatherResult:
        """查询指定地点天气（尚未实现）。

        Args:
            location: 用户输入或解析出的地点名称。

        Raises:
            NotImplementedError: 当前版本未实现高德接口。
        """
        raise NotImplementedError("GaodeAdapter is not implemented yet")
