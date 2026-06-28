"""Agent 运行时模块的公共入口。"""

from legal_assistant.runtime.state import AgentState

__all__ = ["AgentState", "build_agent_graph"]


def __getattr__(name: str):
    if name == "build_agent_graph":
        from legal_assistant.runtime.graph import build_agent_graph

        return build_agent_graph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
