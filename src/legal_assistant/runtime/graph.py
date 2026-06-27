"""Agent 状态图的构建与路由逻辑。

使用 LangGraph 的 ``StateGraph`` 将规划、法律、天气、通用对话等节点
串联成有向图，并根据用户意图决定下一步走向哪个处理节点。
"""

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
    """根据状态中的 ``intent`` 字段决定下一个节点名称。

    这是 LangGraph 条件边的路由函数：返回值必须是 ``add_conditional_edges``
    映射表中已注册的键名。

    Args:
        state: 当前 Agent 状态，至少应包含 ``intent``（由 planner 节点写入）。

    Returns:
        目标节点名：``"legal"``、``"weather"`` 或 ``"general"``。
        若 intent 缺失或无法识别，默认走 ``"general"`` 通用对话分支。
    """
    intent = state.get("intent") or "general"
    if intent == "legal":
        return "legal"
    if intent == "weather":
        return "weather"
    return "general"


def build_agent_graph(deps: RuntimeDeps | None = None):
    """构建并编译完整的 Agent 执行图。

    图的整体流程：
    START → planner（意图分类）→ [legal | weather | general] → save_memory → END

    Args:
        deps: 运行时依赖容器（LLM、检索器、记忆管理等）。
              传入 ``None`` 时会创建默认的 ``RuntimeDeps``，便于测试注入 mock。

    Returns:
        已 ``compile()`` 的可执行图对象，可直接 ``ainvoke(state)`` 运行。
    """
    deps = deps or RuntimeDeps()

    # 以 AgentState 为状态 schema 创建有状态图
    graph = StateGraph(AgentState)

    # 注册各业务节点；工厂函数返回的是 async callable，LangGraph 会异步调用
    graph.add_node("planner", make_planner_node(deps))
    graph.add_node("legal", make_legal_node(deps))
    graph.add_node("weather", make_weather_node(deps))
    graph.add_node("general", make_general_node(deps))
    graph.add_node("save_memory", make_save_memory_node(deps))

    # 固定边：每次请求都从 planner 开始
    graph.add_edge(START, "planner")

    # 条件边：planner 完成后按 intent 分流到三个处理节点之一
    graph.add_conditional_edges(
        "planner",
        route_by_intent,
        {
            "legal": "legal",
            "weather": "weather",
            "general": "general",
        },
    )

    # 三个处理分支汇合后统一写入记忆，再结束
    graph.add_edge("legal", "save_memory")
    graph.add_edge("weather", "save_memory")
    graph.add_edge("general", "save_memory")
    graph.add_edge("save_memory", END)

    return graph.compile()
