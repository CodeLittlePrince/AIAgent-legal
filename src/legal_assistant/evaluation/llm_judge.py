"""基于大语言模型（LLM）的法律回答质量评审。

本模块使用独立的 LLM（默认 DeepSeek）作为「评审员」，对用户问题与助手回答
在相关性、准确性、免责声明等维度进行 1–5 分打分。适用于：

- 离线批量评估生成质量
- 与人工标注对比，校准自动化评分
- 在 A/B 测试或 prompt 迭代后快速对比回答质量

评审 prompt 要求模型仅输出 JSON，解析失败时会尝试从回复中提取 JSON 片段。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from legal_assistant.config import settings

# 评审员 system/user 风格的中文 prompt 模板；占位符由调用方填入
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
    """LLM 评审员对单条回答的打分结果。

    Attributes:
        relevance: 相关性得分，1（完全不相关）到 5（非常切题）。
        accuracy: 准确性得分，1（错误/编造）到 5（准确可靠）。
        disclaimer: 免责声明完整度，1（完全没有）到 5（清晰完整）。
        rationale: 评审员给出的简短理由说明。
    """

    relevance: int
    accuracy: int
    disclaimer: int
    rationale: str = ""


def _parse_judge_json(content: str) -> dict:
    """解析评审 LLM 返回的 JSON 内容。

    先尝试整段解析；若失败则用正则提取第一个 ``{...}`` 片段再解析。
    部分模型会在 JSON 外包裹 markdown 或说明文字，此函数提高容错性。

    Args:
        content: LLM 原始回复文本。

    Returns:
        解析后的字典，应包含 relevance、accuracy、disclaimer 等键。

    Raises:
        json.JSONDecodeError: 无法从内容中解析出有效 JSON。
    """
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def create_judge_llm() -> ChatOpenAI:
    """创建用于评审的 ChatOpenAI 实例。

    使用项目配置中的 DeepSeek 模型、API Key 与 Base URL；
    temperature=0.0 以保证打分稳定、可复现。

    Returns:
        配置好的 ``ChatOpenAI`` 客户端，可用于 ``ainvoke`` 异步调用。
    """
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
    """对一条法律问答进行多维度打分。

    将问题、回答及 API 返回的免责声明填入评审 prompt，调用 LLM 获取 JSON 评分，
    并将各维度分数钳制在 1–5 范围内（解析失败或越界时回退到边界值）。

    Args:
        question: 用户原始问题。
        answer: 助手生成的回答正文。
        disclaimer: API 返回的免责声明文本；为 None 时在 prompt 中显示「(无)」。
        llm: 可选的自定义 LLM 实例；为 None 时使用 ``create_judge_llm()`` 默认配置。

    Returns:
        包含 relevance、accuracy、disclaimer 及 rationale 的 ``JudgeScores``。
    """
    model = llm or create_judge_llm()
    prompt = JUDGE_PROMPT.format(
        question=question,
        answer=answer,
        disclaimer=disclaimer or "(无)",
    )
    response = await model.ainvoke([HumanMessage(content=prompt)])
    payload = _parse_judge_json(str(response.content))

    def _clamp_score(value: object) -> int:
        """将任意值转为 1–5 整数；无法转换时默认为 1。"""
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
