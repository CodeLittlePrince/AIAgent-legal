"""法律知识向量检索器。

两阶段检索：向量库粗筛召回较多候选，再由 Cross-Encoder rerank 精排取 top_k。
粗筛阶段不做严格相似度阈值过滤，避免口语化问法在首轮被全部丢弃。
"""

from __future__ import annotations

from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from legal_assistant.config import settings
from legal_assistant.knowledge.documents import RetrievedDoc
from legal_assistant.knowledge.ingest import LEGAL_COLLECTION, _get_chroma_client
from legal_assistant.knowledge.reranker import DocumentReranker

# 向后兼容：外部仍可从 retriever 导入 RetrievedDoc
__all__ = ["LegalRetriever", "RetrievedDoc"]


class LegalRetriever:
    """连接 Chroma 法律知识库并执行「粗筛 + rerank」检索。"""

    def __init__(
        self,
        chroma_host: str | None = None,
        chroma_port: int | None = None,
        *,
        reranker: DocumentReranker | None = None,
    ):
        """初始化检索器：加载嵌入模型、连接 Chroma 并构建 VectorStoreIndex。

        Args:
            chroma_host: Chroma 服务主机，可选。
            chroma_port: Chroma 服务端口，可选。
            reranker: 可选注入的精排器，便于单测 mock。
        """
        self._chroma_host = chroma_host
        self._chroma_port = chroma_port
        self._reranker = reranker or DocumentReranker()
        embed_model = HuggingFaceEmbedding(model_name=settings.embedding_model)
        Settings.embed_model = embed_model
        client = _get_chroma_client(chroma_host, chroma_port)
        collection = client.get_or_create_collection(LEGAL_COLLECTION)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        self._index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=embed_model
        )
        self._retriever = self._index.as_retriever(similarity_top_k=settings.rag_coarse_top_k)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedDoc]:
        """粗筛 + rerank 检索最相关的法条片段。

        Args:
            query: 用户自然语言问题。
            top_k: 最终返回条数，默认 ``settings.rag_final_top_k``。

        Returns:
            经 rerank 后的 ``RetrievedDoc`` 列表，按相关度降序。
        """
        final_k = top_k if top_k is not None else settings.rag_final_top_k
        nodes = self._retriever.retrieve(query)
        candidates = self._coarse_candidates(nodes)
        return self._reranker.rerank(query, candidates, top_k=final_k)

    @staticmethod
    def _coarse_candidates(nodes) -> list[RetrievedDoc]:
        """粗筛：保留向量召回结果，去重，不做相似度阈值过滤。"""
        seen: set[tuple[str, str]] = set()
        candidates: list[RetrievedDoc] = []
        for node in nodes:
            source = node.metadata.get("file_name") or node.metadata.get("filename") or "unknown"
            text = node.get_content().strip()
            key = (source, text)
            if not text or key in seen:
                continue
            seen.add(key)
            candidates.append(
                RetrievedDoc(source=source, text=text, score=node.score or 0.0)
            )
        return candidates

    def warmup(self) -> None:
        """预加载 embedding 连接与 rerank 模型。"""
        self._reranker.warmup()
