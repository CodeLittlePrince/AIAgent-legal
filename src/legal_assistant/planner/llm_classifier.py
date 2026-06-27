"""基于 LLM 的意图分类器（规则低置信度时的兜底）。

使用较便宜的 DeepSeek 模型做结构化 JSON 分类，避免关键词冲突或口语化
表达被误判为 general。
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from legal_assistant.config import settings
from legal_assistant.planner.intent import Intent, analyze_by_rules
from legal_assistant.planner.router import LLMClassifier, PlanResult

_VALID_INTENTS: frozenset[str] = frozenset({"legal", "weather", "general"})

_CLASSIFIER_SYSTEM = """你是法律助手系统的意图分类器。根据用户最新消息（必要时参考简短对话历史），判断应路由到哪种能力：

- legal：法律咨询、法条、合同、劳动、刑事、侵权、赔偿等需要检索法条的问题
- weather：天气、气温、降雨、预报等；若能识别中国城市，填入 location
- general：闲聊、泛知识、与法律/天气无关，且不属于上述两类

规则：
1. 只输出一行 JSON，不要 markdown 或解释
2. 格式：{"intent":"legal|weather|general","confidence":0.0-1.0,"location":null或"城市名"}
3. 若一句话包含多个话题，选择用户更关心、应优先处理的主意图
4. confidence 表示你对 intent 判断的确信程度（0~1）
"""


def create_classifier_llm() -> ChatOpenAI:
    """创建用于意图分类的 DeepSeek 客户端（低温、短输出）。"""
    return ChatOpenAI(
        model=settings.deepseek_classifier_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0,
        max_tokens=128,
        http_async_client=httpx.AsyncClient(trust_env=False),
    )


def _format_history(history: list[dict[str, Any]], *, limit: int = 4) -> str:
    """将最近几轮对话格式化为 prompt 片段。"""
    if not history:
        return "（无历史）"
    lines: list[str] = []
    for message in history[-limit:]:
        role = message.get("role", "user")
        content = message.get("content", "")
        if isinstance(content, str) and content.strip():
            lines.append(f"{role}: {content.strip()}")
    return "\n".join(lines) if lines else "（无历史）"


def parse_classifier_response(raw: str) -> PlanResult | None:
    """从 LLM 原始输出解析 ``PlanResult``；失败返回 ``None``。"""
    text = raw.strip()
    if not text:
        return None

    candidates = [text]
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        candidates.insert(0, fence.group(1).strip())
    brace = re.search(r"\{[\s\S]*\}", text)
    if brace:
        candidates.append(brace.group(0))

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue

        intent = str(payload.get("intent", "")).strip().lower()
        if intent not in _VALID_INTENTS:
            continue

        try:
            confidence = float(payload.get("confidence", 0.85))
        except (TypeError, ValueError):
            confidence = 0.85
        confidence = max(0.0, min(1.0, confidence))

        location_raw = payload.get("location")
        location = None
        if isinstance(location_raw, str) and location_raw.strip():
            location = location_raw.strip()

        return PlanResult(
            intent=intent,  # type: ignore[arg-type]
            confidence=confidence,
            location=location if intent == "weather" else None,
        )

    return None


def make_llm_classifier(llm: BaseChatModel | None = None) -> LLMClassifier:
    """绑定 LLM 实例，返回符合 ``router.classify`` 签名的异步分类函数。"""

    async def classify_with_llm(message: str, history: list[dict[str, Any]]) -> PlanResult:
        model = llm or create_classifier_llm()
        user_prompt = (
            f"对话历史（最近几轮）：\n{_format_history(history)}\n\n"
            f"用户最新消息：{message}\n"
        )
        try:
            response = await model.ainvoke(
                [
                    SystemMessage(content=_CLASSIFIER_SYSTEM),
                    HumanMessage(content=user_prompt),
                ]
            )
            parsed = parse_classifier_response(str(response.content))
            if parsed is not None:
                return parsed
        except Exception:
            pass

        # LLM 不可用或解析失败：回退规则结果，避免阻断主流程
        intent, confidence, location = analyze_by_rules(message)
        return PlanResult(intent=intent, confidence=confidence, location=location)

    return classify_with_llm
