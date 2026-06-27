"""Agent 运行时状态定义。

``AgentState`` 是 LangGraph 图在执行过程中传递的共享状态字典。
每个节点读取当前状态、返回部分字段的更新，LangGraph 会自动合并到状态中。
"""

from collections.abc import Awaitable
from typing import Any, Protocol, TypedDict

from legal_assistant.knowledge.retriever import RetrievedDoc
from legal_assistant.tools.base import WeatherResult


class AgentState(TypedDict, total=False):
    """Agent 图执行时的共享状态。

    使用 ``TypedDict`` 且 ``total=False``，表示所有字段均为可选：
    节点只需返回它实际修改的字段，未出现的字段保持不变。

    Attributes:
        session_id: 当前对话会话的唯一标识，用于持久化记忆。
        messages: 对话历史，每项为 ``{"role": "user"|"assistant"|"system", "content": "..."}``。
        intent: 规划节点识别出的用户意图，如 ``"legal"``、``"weather"``、``"general"``。
        location: 从用户消息中解析出的地点（主要用于天气查询）。
        retrieved_docs: 法律节点检索到的相关文档列表。
        tool_result: 天气工具返回的结构化结果。
        answer: 最终要展示给用户的助手回复文本。
        citations: 法律回答附带的引用来源列表。
        error: 若某节点执行失败，此处记录错误信息字符串。
    """

    session_id: str
    messages: list[dict[str, Any]]
    intent: str | None
    location: str | None
    retrieved_docs: list[RetrievedDoc] | None
    tool_result: WeatherResult | None
    answer: str | None
    citations: list[dict[str, str]] | None
    error: str | None


class AgentNode(Protocol):
    """LangGraph 节点 callable 的类型约定。

    使用 ``Protocol`` 而非 ``Callable[[AgentState], ...]``，以便静态类型检查器
    将第一个参数识别为具名参数 ``state``，与 LangGraph ``StateNode`` 签名一致。
    """

    def __call__(self, state: AgentState) -> Awaitable[dict[str, Any]]: ...
