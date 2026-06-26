"""Agent E2E benchmark: multi-turn tasks, metrics, and reporting."""

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

__all__ = [
    "BenchmarkReport",
    "BenchmarkRunner",
    "TaskRunResult",
    "TurnResult",
    "compute_metrics",
    "write_report",
]
