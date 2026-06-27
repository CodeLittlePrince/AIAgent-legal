"""基于规则的意图识别模块。

在用户消息中匹配关键词，快速判断意图类型：
法律咨询（legal）、天气查询（weather）或通用对话（general）。

这是 Planner 的第一层：速度快、不消耗 LLM 调用，适合高置信度场景。
"""

from __future__ import annotations

import re
from typing import Literal

# 三种意图的字面量类型，便于类型检查和 IDE 提示
Intent = Literal["legal", "weather", "general"]

# 法律相关关键词：命中任一即倾向 legal 意图
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
    "犯法",
    "犯罪",
    "打人",
    "打架",
    "殴打",
    "伤害",
    "坐牢",
    "刑罚",
    "量刑",
    "涉嫌",
]

# 天气相关关键词
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

# 寒暄类短句关键词（配合长度判断，避免误判长文本）
GENERAL_KEYWORDS = [
    "你好",
    "您好",
    "hello",
    "hi",
    "谢谢",
    "再见",
    "帮助",
]

# 常见城市名，用于从天气类问题中提取地点
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

# 规则明确匹配时的置信度（供 router 决定是否跳过 LLM）
HIGH_CONFIDENCE = 0.9
# 无法明确分类时的较低置信度（router 可能再调用 LLM 复核）
AMBIGUOUS_CONFIDENCE = 0.5


def _count_matches(text: str, keywords: list[str]) -> int:
    """统计文本中命中了多少个关键词（简单子串包含，非分词）。

    Args:
        text: 用户消息。
        keywords: 待匹配的关键词列表。

    Returns:
        命中数量。
    """
    return sum(1 for keyword in keywords if keyword in text)


def _extract_location(text: str) -> str | None:
    """从天气类问题中提取城市或地点名称。

    优先匹配预定义城市列表；否则用正则匹配「XX的天气」类句式。

    Args:
        text: 用户消息。

    Returns:
        地点名称；无法提取时返回 None。
    """
    for city in CITIES:
        if city in text:
            return city

    # 例如「杭州的天气」「北京今天天气」
    match = re.search(r"([\u4e00-\u9fff]{2,4}?)(?:的)?(?:今天|明天|后天)?天气", text)
    if match:
        location = match.group(1)
        # 排除时间词被误识别为地名
        if location not in {"今天", "明天", "后天", "现在", "目前"}:
            return location

    return None


def analyze_by_rules(msg: str) -> tuple[Intent, float, str | None]:
    """对消息做规则分析，返回意图、置信度和可选地点。

    优先级：法律 vs 天气冲突 → 降为 general；
    否则 legal > weather > 短寒暄 general > 默认 general。

    Args:
        msg: 用户输入的原始消息。

    Returns:
        (intent, confidence, location) 三元组；
        location 仅在 weather 意图且成功提取时有值。
    """
    text = msg.strip()
    legal_count = _count_matches(text, LEGAL_KEYWORDS)
    weather_count = _count_matches(text, WEATHER_KEYWORDS)
    general_count = _count_matches(text, GENERAL_KEYWORDS)

    # 只有疑似天气问题时才尝试提取地点，减少无效正则开销
    location = _extract_location(text) if weather_count > 0 else None

    # 同时命中法律与天气关键词：语义模糊，交给 general 或后续 LLM
    if legal_count > 0 and weather_count > 0:
        return "general", AMBIGUOUS_CONFIDENCE, None

    if legal_count > 0:
        return "legal", HIGH_CONFIDENCE, None

    if weather_count > 0:
        return "weather", HIGH_CONFIDENCE, location

    # 极短寒暄（如「你好」）高置信归为 general
    if general_count > 0 and len(text) <= 10:
        return "general", HIGH_CONFIDENCE, None

    return "general", AMBIGUOUS_CONFIDENCE, None


def classify_by_rules(msg: str) -> Intent:
    """仅返回意图类型，忽略置信度与地点（简化 API）。

    Args:
        msg: 用户消息。

    Returns:
        "legal"、"weather" 或 "general"。
    """
    intent, _, _ = analyze_by_rules(msg)
    return intent
