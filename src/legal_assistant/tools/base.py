from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class WeatherResult:
    location: str
    temperature: float | None
    conditions: str
    forecast_summary: str
    raw_source: dict[str, Any]


class WeatherAdapter(Protocol):
    async def get_weather(self, location: str) -> WeatherResult: ...
