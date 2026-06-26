import pytest

from legal_assistant.evaluation.rag_metrics import (
    GoldenCase,
    compute_recall_at_k,
    is_recall_hit,
    load_golden_cases,
)
from legal_assistant.knowledge.retriever import RetrievedDoc, LegalRetriever

MIN_RECALL_AT_5 = 0.7


def test_load_golden_cases_has_expected_count():
    cases = load_golden_cases()
    assert len(cases) >= 20


def test_golden_cases_have_required_fields():
    cases = load_golden_cases()
    for case in cases:
        assert case.id
        assert case.question
        assert case.expected_source


def test_is_recall_hit_matches_expected_source_substring():
    sources = ["中华人民共和国民法典.md", "中华人民共和国劳动法.md"]
    assert is_recall_hit(sources, "民法典", k=5)
    assert not is_recall_hit(sources, "消费者", k=5)


def test_compute_recall_at_k_with_mock_retriever():
    cases = [
        GoldenCase(id="a", question="q1", expected_source="民法典"),
        GoldenCase(id="b", question="q2", expected_source="消费者"),
    ]

    def retrieve(question: str, top_k: int = 5) -> list[RetrievedDoc]:
        if question == "q1":
            return [RetrievedDoc(source="中华人民共和国民法典.md", text="", score=0.9)]
        return [RetrievedDoc(source="中华人民共和国劳动法.md", text="", score=0.9)]

    assert compute_recall_at_k(cases, retrieve, k=5) == 0.5


@pytest.mark.slow
def test_rag_recall_at_5(ingested_retriever: LegalRetriever):
    cases = load_golden_cases()

    def retrieve(question: str, top_k: int = 5) -> list[RetrievedDoc]:
        return ingested_retriever.retrieve(question, top_k=top_k)

    recall = compute_recall_at_k(cases, retrieve, k=5)
    assert recall >= MIN_RECALL_AT_5, (
        f"recall@5={recall:.2%} below threshold {MIN_RECALL_AT_5:.0%}"
    )
