from legal_assistant.tools.base import WeatherResult


class QWeatherAdapter:
    async def get_weather(self, location: str) -> WeatherResult:
        raise NotImplementedError("QWeatherAdapter is not implemented yet")
