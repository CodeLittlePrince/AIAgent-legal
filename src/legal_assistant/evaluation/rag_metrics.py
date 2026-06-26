from dataclasses import dataclass
from pathlib import Path

import yaml

from legal_assistant.knowledge.retriever import RetrievedDoc

GOLDEN_CASES_PATH = Path(__file__).resolve().parent / "golden_cases.yaml"


@dataclass(frozen=True)
class GoldenCase:
    id: str
    question: str
    expected_source: str


def load_golden_cases(path: Path | None = None) -> list[GoldenCase]:
    cases_path = path or GOLDEN_CASES_PATH
    with cases_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return [
        GoldenCase(
            id=entry["id"],
            question=entry["question"],
            expected_source=entry["expected_source"],
        )
        for entry in payload["cases"]
    ]


def is_recall_hit(sources: list[str], expected_source: str, k: int = 5) -> bool:
    for source in sources[:k]:
        if expected_source in source:
            return True
    return False


def compute_recall_at_k(
    cases: list[GoldenCase],
    retrieve_fn,
    k: int = 5,
) -> float:
    if not cases:
        return 0.0
    hits = sum(
        1
        for case in cases
        if is_recall_hit(
            [doc.source for doc in retrieve_fn(case.question, top_k=k)],
            case.expected_source,
            k=k,
        )
    )
    return hits / len(cases)


def docs_to_sources(docs: list[RetrievedDoc]) -> list[str]:
    return [doc.source for doc in docs]
