"""法律知识向量检索器。

基于 Chroma 中已入库的法条片段，对用户问题进行语义相似度检索，
过滤低分结果后返回带来源与得分的结构化文档列表。
"""

from dataclasses import dataclass

import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from legal_assistant.config import settings
from legal_assistant.knowledge.ingest import LEGAL_COLLECTION, _get_chroma_client

# 相似度得分低于此阈值的检索结果被丢弃，减少无关法条干扰
SCORE_THRESHOLD = 0.5


@dataclass
class RetrievedDoc:
    """单条检索结果的数据结构。

    Attributes:
        source: 法条片段来源（通常为 Markdown 文件名）。
        text: 片段正文内容。
        score: 与查询的相似度得分（越高越相关）。
    """

    source: str
    text: str
    score: float


class LegalRetriever:
    """连接 Chroma 法律知识库并执行向量检索的封装类。"""

    def __init__(self, chroma_host: str | None = None, chroma_port: int | None = None):
        """初始化检索器：加载嵌入模型、连接 Chroma 并构建 VectorStoreIndex。

        Args:
            chroma_host: Chroma 服务主机，可选。
            chroma_port: Chroma 服务端口，可选。
        """
        self._chroma_host = chroma_host
        self._chroma_port = chroma_port
        embed_model = HuggingFaceEmbedding(model_name=settings.embedding_model)
        Settings.embed_model = embed_model
        client = _get_chroma_client(chroma_host, chroma_port)
        collection = client.get_or_create_collection(LEGAL_COLLECTION)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        # 从已有向量库恢复索引，而非重新嵌入文档
        self._index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=embed_model
        )
        self._retriever = self._index.as_retriever(similarity_top_k=5)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        """根据用户问题检索最相关的法条片段。

        Args:
            query: 用户自然语言问题。
            top_k: 最多返回的结果条数（在内部 retriever 结果上再次截断）。

        Returns:
            通过得分阈值过滤后的 ``RetrievedDoc`` 列表，按相关度排序。
        """
        nodes = self._retriever.retrieve(query)
        results: list[RetrievedDoc] = []
        for node in nodes[:top_k]:
            score = node.score or 0.0
            if score < SCORE_THRESHOLD:
                continue
            # 兼容不同入库路径下的元数据字段名
            source = node.metadata.get("file_name") or node.metadata.get("filename") or "unknown"
            results.append(
                RetrievedDoc(source=source, text=node.get_content(), score=score)
            )
        return results
