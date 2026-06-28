"""warmup_rag_models 单元测试。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from legal_assistant.knowledge.warmup import warmup_rag_models


@patch("legal_assistant.knowledge.warmup.LegalRetriever")
def test_warmup_rag_models_loads_retriever_and_reranker(mock_retriever_cls):
    retriever = MagicMock()
    mock_retriever_cls.return_value = retriever

    result = warmup_rag_models()

    mock_retriever_cls.assert_called_once()
    retriever.warmup.assert_called_once()
    assert result is retriever
