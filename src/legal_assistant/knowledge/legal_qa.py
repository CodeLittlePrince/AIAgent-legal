from legal_assistant.config import settings
from legal_assistant.knowledge.retriever import RetrievedDoc

EXCERPT_MAX_LEN = 200


def build_legal_prompt(query: str, docs: list[RetrievedDoc]) -> str:
    if not docs:
        return (
            "你是中国法律助手。用户问题如下，但未检索到可靠法条片段。\n"
            f"用户问题：{query}\n"
            "请明确告知用户未找到可靠法条，建议咨询执业律师，不要编造法律依据。"
        )

    context_blocks = []
    for index, doc in enumerate(docs, start=1):
        context_blocks.append(
            f"[{index}] 来源: {doc.source}\n内容: {doc.text.strip()}"
        )
    context = "\n\n".join(context_blocks)

    return (
        "你是中国法律助手。请仅基于以下检索到的法条片段回答用户问题。\n"
        "若片段不足以回答，请明确说明并建议咨询执业律师，不要编造。\n\n"
        f"检索片段：\n{context}\n\n"
        f"用户问题：{query}"
    )


def format_citations(docs: list[RetrievedDoc]) -> list[dict[str, str]]:
    citations: list[dict[str, str]] = []
    for doc in docs:
        excerpt = doc.text.strip()
        if len(excerpt) > EXCERPT_MAX_LEN:
            excerpt = excerpt[:EXCERPT_MAX_LEN] + "..."
        citations.append({"source": doc.source, "excerpt": excerpt})
    return citations


def append_disclaimer(answer: str) -> str:
    disclaimer = settings.legal_disclaimer.strip()
    if disclaimer and disclaimer not in answer:
        return f"{answer.rstrip()}\n\n{disclaimer}"
    return answer
