import re

import pytest

from legal_assistant.tools.registry import get_weather_adapter
from legal_assistant.tools.weather.gaode import GaodeAdapter
from legal_assistant.tools.weather.open_meteo import OpenMeteoAdapter
from legal_assistant.tools.weather.qweather import QWeatherAdapter

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

GEOCODING_RESPONSE = {
    "results": [
        {
            "id": 1816670,
            "name": "北京",
            "latitude": 39.9075,
            "longitude": 116.39723,
            "country": "中国",
            "admin1": "北京市",
        }
    ]
}

FORECAST_RESPONSE = {
    "current": {
        "temperature_2m": 22.5,
        "weather_code": 2,
    },
    "daily": {
        "time": ["2025-06-25", "2025-06-26", "2025-06-27"],
        "weather_code": [2, 3, 61],
        "temperature_2m_max": [28.0, 26.0, 24.0],
        "temperature_2m_min": [18.0, 17.0, 16.0],
    },
}


@pytest.mark.asyncio
async def test_open_meteo_beijing(httpx_mock):
    httpx_mock.add_response(
        url=re.compile(rf"^{re.escape(GEOCODING_URL)}"),
        json=GEOCODING_RESPONSE,
    )
    httpx_mock.add_response(
        url=re.compile(rf"^{re.escape(FORECAST_URL)}"),
        json=FORECAST_RESPONSE,
    )

    adapter = OpenMeteoAdapter()
    result = await adapter.get_weather("北京")

    assert result.location == "北京, 北京市, 中国"
    assert result.temperature == 22.5
    assert result.conditions == "局部多云"
    assert "2025-06-25" in result.forecast_summary
    assert result.raw_source["geocoding"] == GEOCODING_RESPONSE
    assert result.raw_source["forecast"] == FORECAST_RESPONSE


@pytest.mark.asyncio
async def test_open_meteo_location_not_found(httpx_mock):
    httpx_mock.add_response(
        url=re.compile(rf"^{re.escape(GEOCODING_URL)}"),
        json={"results": []},
    )

    adapter = OpenMeteoAdapter()
    with pytest.raises(ValueError, match="未找到地点"):
        await adapter.get_weather("不存在的地点")


def test_get_weather_adapter_open_meteo():
    adapter = get_weather_adapter("open_meteo")
    assert isinstance(adapter, OpenMeteoAdapter)


def test_get_weather_adapter_unknown_provider():
    with pytest.raises(ValueError, match="Unknown weather provider"):
        get_weather_adapter("invalid")


@pytest.mark.asyncio
async def test_qweather_stub_raises():
    adapter = QWeatherAdapter()
    with pytest.raises(NotImplementedError):
        await adapter.get_weather("北京")


@pytest.mark.asyncio
async def test_gaode_stub_raises():
    adapter = GaodeAdapter()
    with pytest.raises(NotImplementedError):
        await adapter.get_weather("北京")
