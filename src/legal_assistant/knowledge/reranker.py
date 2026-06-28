"""法律检索精排：对粗筛候选用法条 Cross-Encoder 重新打分。"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from legal_assistant.config import settings
from legal_assistant.knowledge.documents import RetrievedDoc


class DocumentReranker:
    """基于 Cross-Encoder 的文档精排器（模型懒加载）。"""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or settings.rag_rerank_model
        self._model: CrossEncoder | None = None

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self._model_name)
        return self._model

    def warmup(self) -> None:
        """预加载 Cross-Encoder（触发 HuggingFace 下载与权重加载）。"""
        self._get_model()

    def rerank(self, query: str, docs: list[RetrievedDoc], top_k: int) -> list[RetrievedDoc]:
        """按 query-doc 相关性精排，返回 top_k 条（score 为 rerank 分数）。"""
        if not docs:
            return []
        if len(docs) <= top_k:
            return self._score_and_sort(query, docs)[:top_k]

        scored = self._score_and_sort(query, docs)
        if settings.rag_rerank_score_threshold is not None:
            scored = [
                doc for doc in scored if doc.score >= settings.rag_rerank_score_threshold
            ]
        return scored[:top_k]

    def _score_and_sort(self, query: str, docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
        pairs = [(query, doc.text) for doc in docs]
        scores = self._get_model().predict(pairs, show_progress_bar=False)
        ranked = [
            RetrievedDoc(source=doc.source, text=doc.text, score=float(score))
            for doc, score in zip(docs, scores, strict=True)
        ]
        ranked.sort(key=lambda doc: doc.score, reverse=True)
        return ranked
