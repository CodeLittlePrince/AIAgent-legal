import pytest

from legal_assistant.knowledge.ingest import ingest_legal_documents
from legal_assistant.knowledge.retriever import LegalRetriever


@pytest.fixture(scope="session")
def ingested_retriever() -> LegalRetriever:
    ingest_legal_documents(chroma_host="localhost")
    return LegalRetriever(chroma_host="localhost")
