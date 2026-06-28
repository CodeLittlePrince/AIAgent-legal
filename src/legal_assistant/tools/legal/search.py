"""法律 RAG 检索 Tool（封装 ``LegalRetriever`` 供 Agent Tool Calling 使用）。"""

from __future__ import annotations

from langchain_core.tools import StructuredTool

from legal_assistant.knowledge.retriever import RetrievedDoc
from legal_assistant.runtime.deps import RuntimeDeps
from legal_assistant.tools.constants import LEGAL_SEARCH_TOOL
from legal_assistant.tools.context import AgentToolContext


def _format_retrieved_docs(docs: list[RetrievedDoc]) -> str:
    blocks = [
        f"[{index}] 来源: {doc.source}\n内容: {doc.text.strip()}"
        for index, doc in enumerate(docs, start=1)
    ]
    return "\n\n".join(blocks)


def create_legal_search_tool(deps: RuntimeDeps, ctx: AgentToolContext) -> StructuredTool:
    """构建 ``search_legal_knowledge`` StructuredTool。"""

    async def search_legal_knowledge(query: str) -> str:
        """检索中国法律文档片段，用于回答法律咨询。输入应为用户的完整法律问题。"""
        ctx.tools_used.add(LEGAL_SEARCH_TOOL)
        docs = deps.get_retriever().retrieve(query)
        ctx.retrieved_docs = docs
        if not docs:
            return "未检索到可靠法条片段，请明确告知用户并建议咨询执业律师。"
        return _format_retrieved_docs(docs)

    return StructuredTool.from_function(
        coroutine=search_legal_knowledge,
        name=LEGAL_SEARCH_TOOL,
        description="检索法律知识库，回答合同、劳动、刑法等法律咨询前必须先调用。",
    )
