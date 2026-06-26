import pytest

from legal_assistant.knowledge.ingest import ingest_legal_documents
from legal_assistant.knowledge.retriever import LegalRetriever


@pytest.mark.slow
def test_retrieve_probation_period():
    ingest_legal_documents(chroma_host="localhost")
    retriever = LegalRetriever(chroma_host="localhost")
    results = retriever.retrieve("劳动合同试用期最长多久")
    assert results, "Expected at least one retrieval result"
    sources = " ".join(r.source for r in results)
    assert "劳动" in sources
