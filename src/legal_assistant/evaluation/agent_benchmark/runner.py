"""Agent 基准测试执行器：加载任务 YAML、调用聊天 API、校验每轮期望。

``BenchmarkRunner`` 是核心类：它读取 ``tasks.yaml`` 中定义的多轮对话任务，
通过 HTTP（或注入的 ``chat_fn``）调用 ``/api/v1/chat``，并依据每轮的 ``expect``
字段判断意图、引用、免责声明、工具调用、记忆连贯性等是否满足要求。

支持 ``with BenchmarkRunner(...) as runner:`` 上下文管理，自动关闭 HTTP 客户端。
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

# 默认任务定义文件（与本模块同目录下的 tasks.yaml）
DEFAULT_TASKS_PATH = Path(__file__).resolve().parent / "tasks.yaml"


@dataclass
class TurnResult:
    """单轮对话的执行结果与校验信息。

    记录用户输入、API 响应字段、耗时，以及后续 ``_evaluate_turn`` 写入的 checks。

    Attributes:
        turn_index: 在本任务中的轮次序号，从 0 开始。
        user_message: 该轮发送给 API 的用户消息。
        status_code: HTTP 状态码（或 chat_fn 模拟返回的状态码）。
        intent: API 识别的意图（如 legal、weather 等），失败时可能为 None。
        answer: 助手回答正文。
        citations: 引用列表，每项通常为含 source 等键的字典。
        disclaimer: 免责声明文本。
        trace_id: 可观测性/trace 标识，便于在 Langfuse 等系统中关联。
        latency_ms: 该轮请求耗时（毫秒）。
        error: 非 200 或异常时的错误描述。
        expect: 从任务 YAML 复制的该轮期望条件（只读参考）。
        checks: 各期望项的布尔校验结果，由 ``_evaluate_turn`` 填充。
    """

    turn_index: int
    user_message: str
    status_code: int
    intent: str | None = None
    answer: str | None = None
    citations: list[dict[str, str]] = field(default_factory=list)
    disclaimer: str | None = None
    trace_id: str | None = None
    latency_ms: float = 0.0
    error: str | None = None
    expect: dict[str, Any] = field(default_factory=dict)
    checks: dict[str, bool] = field(default_factory=dict)


@dataclass
class TaskRunResult:
    """单个 benchmark 任务（可能含多轮）的完整运行结果。

    Attributes:
        task_id: 任务 ID，来自 YAML。
        task_name: 任务展示名称；缺省时回退为 task_id。
        category: 任务分类标签，便于报告中分组。
        session_id: 会话 ID；首轮会生成 UUID，后续轮次沿用 API 返回的 session_id。
        turns: 各轮 ``TurnResult`` 列表。
        task_success: 是否所有轮次均 HTTP 200 且通过 ``_evaluate_turn`` 校验。
    """

    task_id: str
    task_name: str
    category: str
    session_id: str
    turns: list[TurnResult] = field(default_factory=list)
    task_success: bool = False


# 聊天函数类型：接收 (session_id, message)，返回包含 status_code、answer 等键的字典
ChatFn = Callable[[str | None, str], dict[str, Any]]


class BenchmarkRunner:
    """针对 ``/api/v1/chat`` 端点执行 benchmark 任务。

    可通过 ``chat_fn`` 注入 mock 或自定义客户端，便于单元测试；
    未注入时使用 ``httpx.Client`` 对 ``base_url`` 发 POST 请求。

    Example:
        >>> with BenchmarkRunner(base_url="http://localhost:8000") as runner:
        ...     results = runner.run_all()
    """

    def __init__(
        self,
        *,
        tasks_path: Path | str | None = None,
        base_url: str = "http://localhost:8000",
        chat_fn: ChatFn | None = None,
        http_client: httpx.Client | None = None,
        timeout: float = 60.0,
    ) -> None:
        """初始化 Runner。

        Args:
            tasks_path: 任务 YAML 路径；None 时使用 ``DEFAULT_TASKS_PATH``。
            base_url: API 服务根 URL，末尾斜杠会被去掉。
            chat_fn: 自定义聊天调用；None 时使用内置 HTTP POST。
            http_client: 可复用的 httpx 客户端；None 时在首次请求时懒创建。
            timeout: HTTP 请求超时秒数。
        """
        self.tasks_path = Path(tasks_path) if tasks_path else DEFAULT_TASKS_PATH
        self.base_url = base_url.rstrip("/")
        self.chat_fn = chat_fn
        self._http_client = http_client
        self.timeout = timeout
        # 标记是否由本 Runner 创建客户端，以便 close() 时只关闭自有的连接
        self._owns_client = False

    def load_tasks(self) -> list[dict[str, Any]]:
        """从 ``tasks_path`` 读取 YAML 并返回 ``tasks`` 列表。

        YAML 根节点应包含 ``tasks`` 键；缺失或为空时返回空列表。
        """
        with self.tasks_path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return list(data.get("tasks") or [])

    def _get_client(self) -> httpx.Client:
        """获取 HTTP 客户端；若尚未创建则按 base_url 与 timeout 新建。"""
        if self._http_client is None:
            self._http_client = httpx.Client(base_url=self.base_url, timeout=self.timeout)
            self._owns_client = True
        return self._http_client

    def close(self) -> None:
        """关闭由本 Runner 创建的 HTTP 客户端，释放连接池。"""
        if self._owns_client and self._http_client is not None:
            self._http_client.close()
            self._http_client = None
            self._owns_client = False

    def __enter__(self) -> BenchmarkRunner:
        """支持 ``with`` 语法，进入时不做额外操作。"""
        return self

    def __exit__(self, *args: object) -> None:
        """退出 ``with`` 块时自动关闭客户端。"""
        self.close()

    def _default_chat(self, session_id: str | None, message: str) -> dict[str, Any]:
        """默认实现：POST ``/api/v1/chat`` 并组装统一结果字典。

        成功 (200) 时将响应 JSON 合并进结果；失败时尝试解析 ``detail`` 作为 error。
        始终包含 ``status_code`` 与 ``latency_ms``。
        """
        client = self._get_client()
        payload: dict[str, str] = {"message": message}
        if session_id:
            payload["session_id"] = session_id

        started = time.perf_counter()
        response = client.post("/api/v1/chat", json=payload)
        latency_ms = (time.perf_counter() - started) * 1000

        result: dict[str, Any] = {
            "status_code": response.status_code,
            "latency_ms": latency_ms,
        }
        if response.status_code == 200:
            result.update(response.json())
        else:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            result["error"] = str(detail)
        return result

    def run_task(self, task: dict[str, Any]) -> TaskRunResult:
        """执行单个任务：按 YAML 中的 ``turns`` 顺序逐轮调用 chat 并汇总结果。

        每轮使用同一 ``session_id``（首轮生成 UUID，若 API 返回新 session_id 则更新），
        以保证多轮记忆测试在同一会话内进行。

        Args:
            task: 单条任务 dict，需含 ``id``、``turns`` 等字段。

        Returns:
            ``TaskRunResult``，``task_success`` 为所有轮均 200 且校验通过。
        """
        chat = self.chat_fn or self._default_chat
        session_id = str(uuid.uuid4())
        turns_spec = task.get("turns") or []
        turn_results: list[TurnResult] = []

        for index, turn_spec in enumerate(turns_spec):
            user_message = turn_spec["user"]
            expect = dict(turn_spec.get("expect") or {})
            raw = chat(session_id, user_message)

            # 服务端可能回写 session_id，后续轮次需保持一致
            if raw.get("session_id"):
                session_id = raw["session_id"]

            turn_result = TurnResult(
                turn_index=index,
                user_message=user_message,
                status_code=int(raw.get("status_code", 500)),
                intent=raw.get("intent"),
                answer=raw.get("answer"),
                citations=list(raw.get("citations") or []),
                disclaimer=raw.get("disclaimer"),
                trace_id=raw.get("trace_id"),
                latency_ms=float(raw.get("latency_ms", 0.0)),
                error=raw.get("error"),
                expect=expect,
            )
            turn_results.append(turn_result)

        task_success = all(
            turn.status_code == 200 and _evaluate_turn(turn, turn_results)
            for turn in turn_results
        )
        return TaskRunResult(
            task_id=task["id"],
            task_name=task.get("name", task["id"]),
            category=task.get("category", "unknown"),
            session_id=session_id,
            turns=turn_results,
            task_success=task_success,
        )

    def run_all(self) -> list[TaskRunResult]:
        """加载全部任务并依次执行，返回每个任务的 ``TaskRunResult``。"""
        return [self.run_task(task) for task in self.load_tasks()]


def _evaluate_turn(turn: TurnResult, prior_turns: list[TurnResult]) -> bool:
    """根据 ``turn.expect`` 校验单轮是否满足期望，结果写入 ``turn.checks``。

    支持的 expect 键包括但不限于：
    - ``intent``: 期望意图字符串完全匹配
    - ``citations``: True/False 表示是否应有引用
    - ``disclaimer``: True/False 表示是否应有免责声明
    - ``tool_success``: 天气等工具类意图是否调用成功
    - ``answer_contains_any``: 回答中是否包含任一关键词（不区分大小写）
    - ``memory_coherence`` + ``memory_keywords_any`` + ``memory_context_from_turn``: 记忆连贯性

    若无任何 expect 项，则仅要求 ``status_code == 200``。

    Args:
        turn: 当前轮结果（会被原地修改 checks）。
        prior_turns: 当前任务中已执行的所有轮（含当前轮），用于记忆类校验。

    Returns:
        所有已定义 checks 均为 True 时返回 True；无 checks 时等价于 status 200。
    """
    expect = turn.expect
    if not expect:
        return turn.status_code == 200

    checks: dict[str, bool] = {}

    if "intent" in expect:
        checks["intent"] = turn.intent == expect["intent"]

    if expect.get("citations") is True:
        checks["citations"] = len(turn.citations) > 0
    elif expect.get("citations") is False:
        checks["citations"] = len(turn.citations) == 0

    if expect.get("disclaimer") is True:
        checks["disclaimer"] = turn.disclaimer is not None and bool(turn.disclaimer.strip())
    elif expect.get("disclaimer") is False:
        checks["disclaimer"] = turn.disclaimer is None

    if expect.get("tool_success") is True:
        # 天气工具成功：200 + intent=weather + 非空回答
        checks["tool_success"] = (
            turn.status_code == 200
            and turn.intent == "weather"
            and bool(turn.answer)
        )

    keywords = expect.get("answer_contains_any") or []
    if keywords:
        answer = (turn.answer or "").lower()
        checks["answer_contains_any"] = any(kw.lower() in answer for kw in keywords)

    if expect.get("memory_coherence") is True:
        memory_keywords = expect.get("memory_keywords_any") or []
        context_turn_index = expect.get("memory_context_from_turn")
        context_text = ""
        if context_turn_index is not None and 0 <= context_turn_index < len(prior_turns):
            context_turn = prior_turns[context_turn_index]
            context_text = context_turn.user_message or ""
        answer = turn.answer or ""
        combined = f"{context_text} {answer}"
        checks["memory_coherence"] = any(kw in combined for kw in memory_keywords)

    if expect.get("task_success") is True:
        checks["task_success"] = all(checks.values()) if checks else turn.status_code == 200

    turn.checks = checks
    return all(checks.values()) if checks else turn.status_code == 200
