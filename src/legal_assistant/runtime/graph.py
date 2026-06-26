from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from legal_assistant.runtime.nodes import (
    RuntimeDeps,
    make_general_node,
    make_legal_node,
    make_planner_node,
    make_save_memory_node,
    make_weather_node,
)
from legal_assistant.runtime.state import AgentState


def route_by_intent(state: AgentState) -> str:
    intent = state.get("intent") or "general"
    if intent == "legal":
        return "legal"
    if intent == "weather":
        return "weather"
    return "general"


def build_agent_graph(deps: RuntimeDeps | None = None):
    deps = deps or RuntimeDeps()

    graph = StateGraph(AgentState)
    graph.add_node("planner", make_planner_node(deps))
    graph.add_node("legal", make_legal_node(deps))
    graph.add_node("weather", make_weather_node(deps))
    graph.add_node("general", make_general_node(deps))
    graph.add_node("save_memory", make_save_memory_node(deps))

    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner",
        route_by_intent,
        {
            "legal": "legal",
            "weather": "weather",
            "general": "general",
        },
    )
    graph.add_edge("legal", "save_memory")
    graph.add_edge("weather", "save_memory")
    graph.add_edge("general", "save_memory")
    graph.add_edge("save_memory", END)

    return graph.compile()
