"""Prometheus 指标定义与记录辅助函数。

使用 ``prometheus_client`` 暴露 Counter/Histogram，供 ``/metrics`` 端点
或 Grafana 采集：聊天请求量、延迟、LLM token 用量、工具调用次数等。
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

# 聊天请求总数，按意图（legal/weather/general）与状态（success/error 等）分桶
CHAT_REQUESTS_TOTAL = Counter(
    "chat_requests_total",
    "Total chat requests",
    ["intent", "status"],
)

# 单次聊天请求端到端延迟（秒），Histogram 自动计算分位数
CHAT_LATENCY_SECONDS = Histogram(
    "chat_latency_seconds",
    "Chat request latency in seconds",
)

# LLM token 累计用量，按模型名与方向（input/output）区分
LLM_TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "Total LLM tokens",
    ["model", "direction"],
)

# 外部工具调用次数，按工具名与成功/失败状态区分
TOOL_CALLS_TOTAL = Counter(
    "tool_calls_total",
    "Total tool calls",
    ["tool", "status"],
)

# 小写别名，兼容可能引用旧变量名的代码
chat_requests_total = CHAT_REQUESTS_TOTAL
chat_latency_seconds = CHAT_LATENCY_SECONDS
llm_tokens_total = LLM_TOKENS_TOTAL
tool_calls_total = TOOL_CALLS_TOTAL


def record_chat_request(intent: str, status: str) -> None:
    """将一次聊天请求计入 ``CHAT_REQUESTS_TOTAL``。

    Args:
        intent: 路由意图，如 ``"legal"``、``"weather"``、``"general"``。
        status: 处理结果状态，如 ``"success"``、``"error"``。
    """
    CHAT_REQUESTS_TOTAL.labels(intent=intent, status=status).inc()


def record_chat_latency(seconds: float) -> None:
    """记录一次聊天请求的耗时（秒），写入 Histogram。

    Args:
        seconds: 从收到请求到返回响应的 wall-clock 秒数。
    """
    CHAT_LATENCY_SECONDS.observe(seconds)


def record_llm_tokens(model: str, direction: str, count: int) -> None:
    """累计 LLM token 使用量。

    Args:
        model: 模型标识，如配置中的 DeepSeek 模型名。
        direction: ``"input"`` 或 ``"output"``（prompt / completion）。
        count: 本次增加的 token 数；``count <= 0`` 时不记录，避免无效增量。
    """
    if count > 0:
        LLM_TOKENS_TOTAL.labels(model=model, direction=direction).inc(count)


def record_tool_call(tool: str, status: str) -> None:
    """记录一次工具（如天气 API）调用。

    Args:
        tool: 工具名称或 provider 标识。
        status: ``"success"`` 或 ``"error"`` 等状态标签。
    """
    TOOL_CALLS_TOTAL.labels(tool=tool, status=status).inc()


def get_metrics_content() -> tuple[bytes, str]:
    """生成 Prometheus 文本格式的指标快照，供 HTTP ``/metrics`` 响应使用。

    Returns:
        二元组 ``(body_bytes, content_type)``，其中 content_type 为
        ``CONTENT_TYPE_LATEST``（OpenMetrics/Prometheus 标准类型）。
    """
    return generate_latest(), CONTENT_TYPE_LATEST
