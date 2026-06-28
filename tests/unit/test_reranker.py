"""DocumentReranker 单元测试（不加载真实 Cross-Encoder）。"""

from __future__ import annotations

from legal_assistant.knowledge.documents import RetrievedDoc
from legal_assistant.knowledge.reranker import DocumentReranker


class _FakeCrossEncoder:
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.last_pairs: list[tuple[str, str]] | None = None

    def predict(self, pairs, show_progress_bar=False):
        self.last_pairs = list(pairs)
        return self._scores


def test_rerank_orders_by_score_and_returns_top_k():
    reranker = DocumentReranker(model_name="fake")
    reranker._model = _FakeCrossEncoder([0.2, 0.9, 0.5])  # noqa: SLF001

    docs = [
        RetrievedDoc(source="a.md", text="低分", score=0.9),
        RetrievedDoc(source="b.md", text="高分", score=0.9),
        RetrievedDoc(source="c.md", text="中分", score=0.9),
    ]
    ranked = reranker.rerank("打人法律责任", docs, top_k=2)

    assert len(ranked) == 2
    assert ranked[0].source == "b.md"
    assert ranked[0].score == 0.9
    assert ranked[1].source == "c.md"
    assert reranker._model.last_pairs == [  # noqa: SLF001
        ("打人法律责任", "低分"),
        ("打人法律责任", "高分"),
        ("打人法律责任", "中分"),
    ]


def test_rerank_returns_empty_for_no_candidates():
    reranker = DocumentReranker(model_name="fake")
    assert reranker.rerank("q", [], top_k=5) == []
