"""LegalRetriever 粗筛逻辑单元测试（mock 向量节点与 reranker）。"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from legal_assistant.knowledge.documents import RetrievedDoc
from legal_assistant.knowledge.retriever import LegalRetriever


def _node(source: str, text: str, score: float):
    return SimpleNamespace(
        metadata={"file_name": source},
        score=score,
        get_content=lambda t=text: t,
    )


def test_coarse_candidates_dedupes_without_score_threshold():
    nodes = [
        _node("刑法.md", "故意伤害", 0.46),
        _node("刑法.md", "故意伤害", 0.45),
        _node("民法.md", "侵权责任", 0.44),
    ]
    candidates = LegalRetriever._coarse_candidates(nodes)

    assert len(candidates) == 2
    assert candidates[0].text == "故意伤害"
    assert candidates[0].score == 0.46


def test_retrieve_passes_coarse_candidates_to_reranker():
    reranker = MagicMock()
    reranker.rerank.return_value = [
        RetrievedDoc(source="刑法.md", text="第二百三十四条", score=0.88)
    ]

    retriever = LegalRetriever.__new__(LegalRetriever)
    retriever._reranker = reranker
    retriever._retriever = MagicMock()
    retriever._retriever.retrieve.return_value = [
        _node("刑法.md", "第二百三十四条", 0.47),
        _node("民法.md", "侵权责任", 0.46),
    ]

    results = retriever.retrieve("我打人了，需要承担什么法律责任", top_k=3)

    reranker.rerank.assert_called_once()
    call = reranker.rerank.call_args
    assert call.args[0] == "我打人了，需要承担什么法律责任"
    assert len(call.args[1]) == 2
    assert call.kwargs["top_k"] == 3
    assert results[0].text == "第二百三十四条"
