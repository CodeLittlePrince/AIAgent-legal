#!/usr/bin/env python3
"""Rebuild legal knowledge vector index in Chroma."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legal_assistant.knowledge.ingest import ingest_legal_documents


def main() -> None:
    count = ingest_legal_documents()
    print(f"Ingested {count} nodes into collection 'legal_knowledge'")


if __name__ == "__main__":
    main()
