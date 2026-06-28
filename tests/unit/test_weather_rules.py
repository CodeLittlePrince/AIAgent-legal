import pytest

from legal_assistant.planner.weather_rules import detect_weather_route


@pytest.mark.parametrize(
    "message,expected_location",
    [
        ("北京今天天气怎么样？", "北京"),
        ("上海明天会下雨吗", "上海"),
        ("杭州气温多少", "杭州"),
    ],
)
def test_detect_weather_with_keywords(message: str, expected_location: str) -> None:
    route = detect_weather_route(message, [])
    assert route is not None
    assert route.location == expected_location
    assert route.confidence >= 0.85


def test_detect_weather_skips_legal_conflict() -> None:
    assert detect_weather_route("劳动合同和天气有什么关系", []) is None


def test_detect_weather_followup_with_history() -> None:
    history = [
        {"role": "user", "content": "上海今天天气怎么样？"},
        {"role": "assistant", "content": "上海今天晴，22度。"},
    ]
    route = detect_weather_route("那上海呢？", history)
    assert route is not None
    assert route.location == "上海"


def test_detect_weather_followup_sanya_without_na() -> None:
    history = [
        {"role": "user", "content": "杭州天气"},
        {"role": "assistant", "content": "杭州今天多云，气温 26 度。"},
    ]
    route = detect_weather_route("三亚呢", history)
    assert route is not None
    assert route.location == "三亚"


def test_detect_weather_hangzhou_keyword() -> None:
    route = detect_weather_route("杭州天气", [])
    assert route is not None
    assert route.location == "杭州"


def test_general_message_not_weather_route() -> None:
    assert detect_weather_route("你好", []) is None
    assert detect_weather_route("劳动合同试用期多久", []) is None
