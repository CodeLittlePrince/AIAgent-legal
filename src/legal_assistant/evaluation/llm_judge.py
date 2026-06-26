from __future__ import annotations

import json
import re
from dataclasses import dataclass

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from legal_assistant.config import settings

JUDGE_PROMPT = """你是法律问答质量评审员。请根据用户问题和助手回答，对以下维度打分（1-5 整数）：
1. relevance（相关性）：回答是否切题
2. accuracy（准确性）：回答是否准确、无编造
3. disclaimer（免责声明）：是否包含「不构成法律意见」类免责声明（1=完全没有，5=清晰完整）

仅输出 JSON，格式：
{{"relevance": 4, "accuracy": 4, "disclaimer": 5, "rationale": "简短理由"}}

用户问题：{question}

助手回答：
{answer}

免责声明字段（API 返回）：{disclaimer}
"""


@dataclass
class JudgeScores:
    relevance: int
    accuracy: int
    disclaimer: int
    rationale: str = ""


def _parse_judge_json(content: str) -> dict:
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def create_judge_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0.0,
    )


async def judge_legal_answer(
    question: str,
    answer: str,
    disclaimer: str | None = None,
    *,
    llm: ChatOpenAI | None = None,
) -> JudgeScores:
    """Score a legal answer 1-5 on relevance, accuracy, and disclaimer presence."""
    model = llm or create_judge_llm()
    prompt = JUDGE_PROMPT.format(
        question=question,
        answer=answer,
        disclaimer=disclaimer or "(无)",
    )
    response = await model.ainvoke([HumanMessage(content=prompt)])
    payload = _parse_judge_json(str(response.content))

    def _clamp_score(value: object) -> int:
        try:
            score = int(value)
        except (TypeError, ValueError):
            score = 1
        return max(1, min(5, score))

    return JudgeScores(
        relevance=_clamp_score(payload.get("relevance")),
        accuracy=_clamp_score(payload.get("accuracy")),
        disclaimer=_clamp_score(payload.get("disclaimer")),
        rationale=str(payload.get("rationale", "")),
    )
