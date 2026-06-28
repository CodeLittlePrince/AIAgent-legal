"""法律知识检索数据结构。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievedDoc:
    """单条检索结果。

    Attributes:
        source: 法条片段来源（通常为 Markdown 文件名）。
        text: 片段正文内容。
        score: 相关度得分（粗筛为向量相似度，最终结果为 rerank 分数）。
    """

    source: str
    text: str
    score: float
