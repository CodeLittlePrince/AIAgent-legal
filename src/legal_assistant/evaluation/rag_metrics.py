"""RAG（检索增强生成）离线评估指标。

本模块用于评估「检索阶段」的质量：给定标准测试用例（golden cases），
检查检索器返回的文档来源是否包含预期来源，并计算 Recall@K 等指标。

适用场景：
- 修改 embedding 模型或检索策略后，快速对比检索召回率
- 在 CI 或本地脚本中跑回归测试，确保知识库改动未降低检索质量
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from legal_assistant.knowledge.retriever import RetrievedDoc

# 默认 golden cases 文件路径（与本模块同目录下的 golden_cases.yaml）
GOLDEN_CASES_PATH = Path(__file__).resolve().parent / "golden_cases.yaml"


@dataclass(frozen=True)
class GoldenCase:
    """单条标准测试用例（golden case）。

    每条用例包含一个问题，以及检索结果中「应该出现」的预期文档来源标识。
    frozen=True 表示实例创建后不可修改，适合作为只读测试数据。

    Attributes:
        id: 用例唯一标识，便于在日志或报告中引用。
        question: 用户会提出的问题文本。
        expected_source: 预期应被检索到的文档来源（如文件名或路径片段）。
    """

    id: str
    question: str
    expected_source: str


def load_golden_cases(path: Path | None = None) -> list[GoldenCase]:
    """从 YAML 文件加载标准测试用例列表。

    Args:
        path: YAML 文件路径；若为 None，则使用默认的 ``GOLDEN_CASES_PATH``。

    Returns:
        解析后的 ``GoldenCase`` 列表。YAML 中需包含 ``cases`` 键，其值为用例数组。

    Raises:
        FileNotFoundError: 指定路径的文件不存在。
        KeyError: YAML 结构不符合预期（缺少 ``cases`` 或字段名错误）。
    """
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
    """判断在前 K 个检索来源中是否命中预期来源。

    采用子串匹配：只要 ``expected_source`` 出现在某个 ``source`` 字符串中，
    即视为命中（不要求完全一致）。

    Args:
        sources: 检索返回的文档来源列表，通常按相关度排序。
        expected_source: 期望命中的来源标识（子串）。
        k: 只检查排名前 k 的结果，默认 5（对应 Recall@5）。

    Returns:
        若前 k 个来源中至少有一个包含 ``expected_source``，返回 True；否则 False。
    """
    for source in sources[:k]:
        if expected_source in source:
            return True
    return False


def compute_recall_at_k(
    cases: list[GoldenCase],
    retrieve_fn,
    k: int = 5,
) -> float:
    """对一批 golden cases 计算 Recall@K。

    对每条用例调用 ``retrieve_fn(question, top_k=k)`` 获取检索结果，
    统计命中预期来源的用例比例。

    Args:
        cases: 标准测试用例列表。
        retrieve_fn: 检索函数，签名为 ``(question: str, top_k: int) -> list[RetrievedDoc]``。
        k: 评估时考虑的 top-k 检索结果数量，默认 5。

    Returns:
        命中用例数 / 总用例数。若 ``cases`` 为空，返回 0.0（避免除零）。
    """
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
    """将 ``RetrievedDoc`` 列表转换为来源字符串列表。

    便于与 ``is_recall_hit`` 等只接受 ``list[str]`` 的函数配合使用。

    Args:
        docs: 检索器返回的文档对象列表。

    Returns:
        各文档的 ``source`` 字段组成的列表，顺序与输入一致。
    """
    return [doc.source for doc in docs]
