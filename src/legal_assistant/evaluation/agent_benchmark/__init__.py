"""Agent 端到端（E2E）基准测试子包。

本包用于对法律助手进行多轮对话场景的自动化测试，包括：

- **BenchmarkRunner**：从 YAML 加载任务，按轮次调用聊天 API 并校验期望
- **指标与报告**：汇总任务成功率、意图准确率、引用合规率等，并写入 JSON 报告

对外公开的主要符号见 ``__all__``，便于 ``from legal_assistant.evaluation.agent_benchmark import ...`` 使用。
"""

from legal_assistant.evaluation.agent_benchmark.metrics import (
    BenchmarkReport,
    compute_metrics,
    write_report,
)
from legal_assistant.evaluation.agent_benchmark.runner import (
    BenchmarkRunner,
    TaskRunResult,
    TurnResult,
)

# 包对外 API：仅导出以下名称，避免 ``from pkg import *`` 时引入内部实现细节
__all__ = [
    "BenchmarkReport",
    "BenchmarkRunner",
    "TaskRunResult",
    "TurnResult",
    "compute_metrics",
    "write_report",
]
