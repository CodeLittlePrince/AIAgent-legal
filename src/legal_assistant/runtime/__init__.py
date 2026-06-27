"""Agent 运行时模块的公共入口。

本包负责组装 LangGraph 状态图，并定义 Agent 在各节点间流转时共享的状态结构。
外部代码通常只需从此处导入 ``AgentState`` 与 ``build_agent_graph``。
"""

from legal_assistant.runtime.graph import build_agent_graph
from legal_assistant.runtime.state import AgentState

# 明确对外暴露的符号，供 ``from legal_assistant.runtime import ...`` 使用
__all__ = ["AgentState", "build_agent_graph"]
