"""天气查询的规则识别与地点提取（快速路径，不消耗 LLM 做意图分类）。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

WEATHER_KEYWORDS = [
    "天气",
    "气温",
    "温度",
    "下雨",
    "下雪",
    "刮风",
    "预报",
    "湿度",
    "晴天",
    "阴天",
    "多云",
    "降雨",
    "降雪",
    "风力",
    "雾霾",
    "紫外线",
]

# 与法律关键词冲突时不走天气快速路径，交给 Agent
LEGAL_KEYWORDS = [
    "法律",
    "法规",
    "法条",
    "合同",
    "劳动",
    "诉讼",
    "律师",
    "违法",
    "犯法",
]

CITIES = [
    "北京",
    "上海",
    "广州",
    "深圳",
    "杭州",
    "南京",
    "成都",
    "重庆",
    "武汉",
    "西安",
    "天津",
    "苏州",
    "长沙",
    "郑州",
    "青岛",
    "大连",
    "三亚",
    "厦门",
    "昆明",
    "哈尔滨",
    "沈阳",
    "济南",
    "福州",
    "合肥",
    "南昌",
    "南宁",
    "海口",
    "贵阳",
    "兰州",
    "乌鲁木齐",
    "拉萨",
    "银川",
    "西宁",
    "呼和浩特",
    "石家庄",
    "太原",
    "无锡",
    "宁波",
    "温州",
    "珠海",
    "东莞",
    "佛山",
]

HIGH_CONFIDENCE = 0.9


@dataclass(frozen=True)
class WeatherRoute:
    """规则命中时的天气快速路径结果。"""

    location: str
    confidence: float


def _count_matches(text: str, keywords: list[str]) -> int:
    return sum(1 for keyword in keywords if keyword in text)


def extract_location(text: str) -> str | None:
    """从文本中提取城市或地点名。"""
    normalized = text.strip().rstrip("？?")

    for city in CITIES:
        if city in normalized:
            return city

    match = re.search(r"([\u4e00-\u9fff]{2,4}?)(?:的)?(?:今天|明天|后天)?天气", normalized)
    if match:
        location = match.group(1)
        if location not in {"今天", "明天", "后天", "现在", "目前"}:
            return location

    followup_with_na = re.search(r"那([\u4e00-\u9fff]{2,6})呢", normalized)
    if followup_with_na:
        return followup_with_na.group(1)

    # 「三亚呢」「北京呢」类短追问（不含「那」）
    followup_city = re.match(r"^([\u4e00-\u9fff]{2,6})呢$", normalized)
    if followup_city:
        return followup_city.group(1)

    return None


def _is_weather_followup(text: str) -> bool:
    normalized = text.strip().rstrip("？?")
    if extract_location(normalized):
        return True
    if re.search(r"那.+呢", normalized):
        return True
    return bool(re.match(r"^[\u4e00-\u9fff]{2,6}呢$", normalized))


def _history_has_weather_context(history: list[dict[str, Any]]) -> bool:
    for message in history[-8:]:
        content = message.get("content", "")
        if not isinstance(content, str):
            continue
        if _count_matches(content, WEATHER_KEYWORDS) > 0:
            return True
        if any(token in content for token in ("°C", "℃", "度", "晴", "雨", "多云")):
            return True
    return False


def detect_weather_route(
    message: str,
    history: list[dict[str, Any]] | None = None,
) -> WeatherRoute | None:
    """判断当前轮是否应走天气快速路径。

    命中条件（且无法律关键词冲突）：
    1. 当前句含天气关键词；
    2. 或：对话近期有天气上下文，且当前句为短追问（含城市名或「那XX呢」）。
    """
    text = message.strip()
    if not text:
        return None

    history = history or []
    legal_count = _count_matches(text, LEGAL_KEYWORDS)
    weather_count = _count_matches(text, WEATHER_KEYWORDS)

    if legal_count > 0 and weather_count > 0:
        return None

    if weather_count > 0 and legal_count == 0:
        location = extract_location(text) or text
        return WeatherRoute(location=location, confidence=HIGH_CONFIDENCE)

    if legal_count > 0:
        return None

    if _history_has_weather_context(history) and len(text) <= 20:
        if _is_weather_followup(text):
            location = extract_location(text) or text.strip().rstrip("？?")
            return WeatherRoute(location=location, confidence=HIGH_CONFIDENCE)

    return None
