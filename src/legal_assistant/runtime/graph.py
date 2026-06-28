"""Agent 状态图的构建。

START → route → weather（规则+直调工具）| agent（Tool Calling）→ save_memory → END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from legal_assistant.runtime.deps import RuntimeDeps
from legal_assistant.runtime.nodes import (
    make_agent_node,
    make_route_node,
    make_save_memory_node,
    make_weather_node,
)
from legal_assistant.runtime.state import AgentState


def route_after_classify(state: AgentState) -> str:
    if state.get("route") == "weather":
        return "weather"
    return "agent"


def build_agent_graph(deps: RuntimeDeps | None = None):
    deps = deps or RuntimeDeps()

    graph = StateGraph(AgentState)
    graph.add_node("route", make_route_node(deps))
    graph.add_node("weather", make_weather_node(deps))
    graph.add_node("agent", make_agent_node(deps))
    graph.add_node("save_memory", make_save_memory_node(deps))

    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        route_after_classify,
        {"weather": "weather", "agent": "agent"},
    )
    graph.add_edge("weather", "save_memory")
    graph.add_edge("agent", "save_memory")
    graph.add_edge("save_memory", END)

    return graph.compile()
