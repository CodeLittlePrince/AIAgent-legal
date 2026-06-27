"""法律知识库模块的公共入口。

提供法条文档入库（``ingest_legal_documents``）与向量检索（``LegalRetriever``）能力，
供对话流程中的 RAG（检索增强生成）使用。
"""

from legal_assistant.knowledge.ingest import ingest_legal_documents
from legal_assistant.knowledge.retriever import LegalRetriever, RetrievedDoc

__all__ = ["ingest_legal_documents", "LegalRetriever", "RetrievedDoc"]
