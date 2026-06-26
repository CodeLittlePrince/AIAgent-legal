from __future__ import annotations

import re
from typing import Literal

Intent = Literal["legal", "weather", "general"]

LEGAL_KEYWORDS = [
    "法律",
    "法规",
    "法条",
    "条例",
    "合同",
    "劳动",
    "诉讼",
    "侵权",
    "维权",
    "律师",
    "仲裁",
    "赔偿",
    "罪名",
    "刑法",
    "民法",
    "宪法",
    "法院",
    "判决",
    "起诉",
    "立案",
    "缓刑",
    "拘留",
    "罚款",
    "保密",
    "竞业",
    "试用期",
    "工伤",
    "社保",
    "公积金",
    "离婚",
    "继承",
    "知识产权",
    "专利",
    "商标",
    "著作权",
    "工资",
    "加班",
    "解雇",
    "辞退",
    "离职",
    "违约",
    "违法",
]

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

GENERAL_KEYWORDS = [
    "你好",
    "您好",
    "hello",
    "hi",
    "谢谢",
    "再见",
    "帮助",
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
]

HIGH_CONFIDENCE = 0.9
AMBIGUOUS_CONFIDENCE = 0.5


def _count_matches(text: str, keywords: list[str]) -> int:
    return sum(1 for keyword in keywords if keyword in text)


def _extract_location(text: str) -> str | None:
    for city in CITIES:
        if city in text:
            return city

    match = re.search(r"([\u4e00-\u9fff]{2,4}?)(?:的)?(?:今天|明天|后天)?天气", text)
    if match:
        location = match.group(1)
        if location not in {"今天", "明天", "后天", "现在", "目前"}:
            return location

    return None


def analyze_by_rules(msg: str) -> tuple[Intent, float, str | None]:
    text = msg.strip()
    legal_count = _count_matches(text, LEGAL_KEYWORDS)
    weather_count = _count_matches(text, WEATHER_KEYWORDS)
    general_count = _count_matches(text, GENERAL_KEYWORDS)

    location = _extract_location(text) if weather_count > 0 else None

    if legal_count > 0 and weather_count > 0:
        return "general", AMBIGUOUS_CONFIDENCE, None

    if legal_count > 0:
        return "legal", HIGH_CONFIDENCE, None

    if weather_count > 0:
        return "weather", HIGH_CONFIDENCE, location

    if general_count > 0 and len(text) <= 10:
        return "general", HIGH_CONFIDENCE, None

    return "general", AMBIGUOUS_CONFIDENCE, None


def classify_by_rules(msg: str) -> Intent:
    intent, _, _ = analyze_by_rules(msg)
    return intent
