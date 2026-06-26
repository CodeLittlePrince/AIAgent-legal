from legal_assistant.tools.base import WeatherResult


class GaodeAdapter:
    async def get_weather(self, location: str) -> WeatherResult:
        raise NotImplementedError("GaodeAdapter is not implemented yet")
