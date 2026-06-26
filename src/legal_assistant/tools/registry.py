from legal_assistant.tools.base import WeatherAdapter
from legal_assistant.tools.weather.gaode import GaodeAdapter
from legal_assistant.tools.weather.open_meteo import OpenMeteoAdapter
from legal_assistant.tools.weather.qweather import QWeatherAdapter


def get_weather_adapter(provider: str) -> WeatherAdapter:
    adapters: dict[str, type[WeatherAdapter]] = {
        "open_meteo": OpenMeteoAdapter,
        "qweather": QWeatherAdapter,
        "gaode": GaodeAdapter,
    }
    adapter_cls = adapters.get(provider)
    if adapter_cls is None:
        raise ValueError(f"Unknown weather provider: {provider}")
    return adapter_cls()
