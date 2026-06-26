import pytest

from legal_assistant.planner.intent import classify_by_rules
from legal_assistant.planner.router import PlanResult, classify


@pytest.mark.parametrize(
    "msg,expected",
    [
        ("劳动合同试用期多久", "legal"),
        ("北京今天天气", "weather"),
        ("你好", "general"),
        ("公司拖欠工资怎么办", "legal"),
        ("上海明天会下雨吗", "weather"),
        ("请问人工智能是什么", "general"),
    ],
)
def test_rule_based_intent(msg: str, expected: str) -> None:
    assert classify_by_rules(msg) == expected


@pytest.mark.asyncio
async def test_classify_high_confidence_legal() -> None:
    result = await classify("劳动合同试用期多久", [])

    assert result.intent == "legal"
    assert result.confidence >= 0.85
    assert result.location is None


@pytest.mark.asyncio
async def test_classify_weather_extracts_location() -> None:
    result = await classify("北京今天天气", [])

    assert result.intent == "weather"
    assert result.location == "北京"
    assert result.confidence >= 0.85


@pytest.mark.asyncio
async def test_classify_ambiguous_uses_llm() -> None:
    async def mock_llm(message: str, history: list) -> PlanResult:
        return PlanResult(intent="legal", confidence=0.92, location=None)

    result = await classify("帮我查一下", [], llm_classifier=mock_llm)

    assert result.intent == "legal"
    assert result.confidence == 0.92


@pytest.mark.asyncio
async def test_classify_ambiguous_fallback_without_llm() -> None:
    result = await classify("帮我查一下", [])

    assert result.intent == "general"
    assert result.confidence < 0.85
