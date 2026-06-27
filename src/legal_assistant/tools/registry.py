"""天气适配器的注册与工厂函数。

根据配置项 ``weather_provider`` 的名称，实例化对应的适配器类。
新增供应商时：实现 ``WeatherAdapter`` 协议，并在 ``adapters`` 字典中登记即可。
"""

from legal_assistant.tools.base import WeatherAdapter
from legal_assistant.tools.weather.gaode import GaodeAdapter
from legal_assistant.tools.weather.open_meteo import OpenMeteoAdapter
from legal_assistant.tools.weather.qweather import QWeatherAdapter


def get_weather_adapter(provider: str) -> WeatherAdapter:
    """按 provider 名称创建天气适配器实例。

    Args:
        provider: 配置中的供应商标识，支持 ``"open_meteo"``、``"qweather"``、``"gaode"``。

    Returns:
        已实例化的适配器对象（满足 ``WeatherAdapter`` 协议）。

    Raises:
        ValueError: 当 provider 不在已知列表中时抛出。
    """
    # 字符串键 → 适配器类；值是类本身，调用 ``adapter_cls()`` 得到实例
    adapters: dict[str, type[WeatherAdapter]] = {
        "open_meteo": OpenMeteoAdapter,
        "qweather": QWeatherAdapter,
        "gaode": GaodeAdapter,
    }
    adapter_cls = adapters.get(provider)
    if adapter_cls is None:
        raise ValueError(f"Unknown weather provider: {provider}")
    return adapter_cls()
