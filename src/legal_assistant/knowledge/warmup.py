"""RAG 模型预热：在应用启动或构建阶段预下载/加载 embedding 与 rerank 模型。"""

from __future__ import annotations

from legal_assistant.knowledge.retriever import LegalRetriever


def warmup_rag_models() -> LegalRetriever:
    """构造检索器并预加载 rerank Cross-Encoder。

    ``LegalRetriever`` 初始化时会加载 embedding 模型并连接 Chroma；
    ``warmup()`` 会额外加载 rerank 模型，避免首次用户检索时才下载。
    """
    retriever = LegalRetriever()
    retriever.warmup()
    return retriever
