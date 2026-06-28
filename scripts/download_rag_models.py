#!/usr/bin/env python3
"""预下载 RAG 所需的 HuggingFace 模型（embedding + rerank）。

本地开发可在首次启动 uvicorn 前执行::

    python scripts/download_rag_models.py

Docker 镜像构建时已在 Dockerfile 中预下载，一般无需手动运行。
"""

from __future__ import annotations

from legal_assistant.config import settings
from legal_assistant.knowledge.warmup import warmup_rag_models


def main() -> None:
    print(f"Downloading embedding model: {settings.embedding_model}")
    print(f"Downloading rerank model: {settings.rag_rerank_model}")
    warmup_rag_models()
    print("RAG models ready.")


if __name__ == "__main__":
    main()
