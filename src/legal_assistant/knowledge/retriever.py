from dataclasses import dataclass

import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from legal_assistant.config import settings
from legal_assistant.knowledge.ingest import LEGAL_COLLECTION, _get_chroma_client

SCORE_THRESHOLD = 0.5


@dataclass
class RetrievedDoc:
    source: str
    text: str
    score: float


class LegalRetriever:
    def __init__(self, chroma_host: str | None = None, chroma_port: int | None = None):
        self._chroma_host = chroma_host
        self._chroma_port = chroma_port
        embed_model = HuggingFaceEmbedding(model_name=settings.embedding_model)
        Settings.embed_model = embed_model
        client = _get_chroma_client(chroma_host, chroma_port)
        collection = client.get_or_create_collection(LEGAL_COLLECTION)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        self._index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=embed_model
        )
        self._retriever = self._index.as_retriever(similarity_top_k=5)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        nodes = self._retriever.retrieve(query)
        results: list[RetrievedDoc] = []
        for node in nodes[:top_k]:
            score = node.score or 0.0
            if score < SCORE_THRESHOLD:
                continue
            source = node.metadata.get("file_name") or node.metadata.get("filename") or "unknown"
            results.append(
                RetrievedDoc(source=source, text=node.get_content(), score=score)
            )
        return results
