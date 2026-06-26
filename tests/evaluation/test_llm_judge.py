from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from legal_assistant.config import settings
from legal_assistant.evaluation.llm_judge import JudgeScores, judge_legal_answer


pytestmark = pytest.mark.slow


def _has_deepseek_key() -> bool:
    return bool(os.getenv("DEEPSEEK_API_KEY") or settings.deepseek_api_key)


@pytest.fixture
def sample_legal_qa():
    return {
        "question": "劳动合同试用期最长多久？",
        "answer": (
            "根据《劳动合同法》，劳动合同试用期最长不得超过六个月。"
            "本回答仅供参考，不构成法律意见，具体问题请咨询执业律师。"
        ),
        "disclaimer": settings.legal_disclaimer,
    }


@pytest.mark.asyncio
async def test_judge_legal_answer_with_mock_llm(sample_legal_qa):
    mock_llm = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = (
        '{"relevance": 5, "accuracy": 4, "disclaimer": 5, "rationale": "回答切题且含免责声明"}'
    )
    mock_llm.ainvoke.return_value = mock_response

    scores = await judge_legal_answer(
        sample_legal_qa["question"],
        sample_legal_qa["answer"],
        sample_legal_qa["disclaimer"],
        llm=mock_llm,
    )

    assert isinstance(scores, JudgeScores)
    assert scores.relevance == 5
    assert scores.accuracy == 4
    assert scores.disclaimer == 5
    mock_llm.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.skipif(not _has_deepseek_key(), reason="DEEPSEEK_API_KEY not configured")
async def test_judge_legal_answer_live_deepseek(sample_legal_qa):
    scores = await judge_legal_answer(
        sample_legal_qa["question"],
        sample_legal_qa["answer"],
        sample_legal_qa["disclaimer"],
    )

    assert 1 <= scores.relevance <= 5
    assert 1 <= scores.accuracy <= 5
    assert 1 <= scores.disclaimer <= 5
