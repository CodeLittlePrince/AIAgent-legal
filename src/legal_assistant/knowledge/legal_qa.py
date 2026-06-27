"""法律问答 Prompt 构建与回答后处理。

将检索到的法条片段组装成 LLM 系统提示，并格式化引用、追加免责声明，
确保回答有据可查且符合合规要求。
"""

from legal_assistant.config import settings
from legal_assistant.knowledge.retriever import RetrievedDoc

# 引用摘要中单条 excerpt 的最大字符数，超出部分以省略号截断
EXCERPT_MAX_LEN = 200


def build_legal_prompt(query: str, docs: list[RetrievedDoc]) -> str:
    """根据检索结果构建发给 LLM 的完整提示词。

    有检索结果时：要求模型仅基于给定法条片段回答，不足时明确说明。
    无检索结果时：要求模型告知未找到可靠法条，禁止编造法律依据。

    Args:
        query: 用户原始问题。
        docs: ``LegalRetriever`` 返回的相关法条片段列表。

    Returns:
        可直接传入 LLM 的中文字符串 prompt。
    """
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
    """将检索文档格式化为 API 响应中的引用列表。

    每条引用包含来源文件名与截断后的正文摘要，便于前端展示「依据哪条法条」。

    Args:
        docs: 检索到的法条片段列表。

    Returns:
        元素为 ``{"source": ..., "excerpt": ...}`` 的字典列表。
    """
    citations: list[dict[str, str]] = []
    for doc in docs:
        excerpt = doc.text.strip()
        if len(excerpt) > EXCERPT_MAX_LEN:
            excerpt = excerpt[:EXCERPT_MAX_LEN] + "..."
        citations.append({"source": doc.source, "excerpt": excerpt})
    return citations


def append_disclaimer(answer: str) -> str:
    """在助手回答末尾追加法律免责声明（若配置中存在且尚未包含）。

    Args:
        answer: LLM 生成的原始回答文本。

    Returns:
        可能已追加免责声明的最终回答字符串。
    """
    disclaimer = settings.legal_disclaimer.strip()
    # 避免重复追加相同免责声明
    if disclaimer and disclaimer not in answer:
        return f"{answer.rstrip()}\n\n{disclaimer}"
    return answer
