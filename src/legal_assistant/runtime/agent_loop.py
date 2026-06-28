"""Tool Calling Agent（LangChain ``create_agent``）。"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig, merge_configs

from legal_assistant.config import settings
from legal_assistant.knowledge.legal_qa import append_disclaimer, format_citations
from legal_assistant.runtime.deps import RuntimeDeps
from legal_assistant.tools.builder import build_agent_tools
from legal_assistant.tools.constants import LEGAL_SEARCH_TOOL, WEATHER_FORECAST_TOOL
from legal_assistant.tools.context import AgentToolContext

_AGENT_SYSTEM = f"""你是智能法律助手，可通过工具获取法条与天气数据。

规则（必须遵守）：
1. 法律咨询（合同、劳动、刑法、赔偿、法条等）：必须先调用 {LEGAL_SEARCH_TOOL}，再仅基于返回片段用中文回答；不得编造法条；未调用工具前不得给出法律结论。
2. 天气查询（含多轮追问，如「那上海呢」「三亚呢」等省略主语的续问）：必须先调用 {WEATHER_FORECAST_TOOL} 获取数据，再基于工具结果用中文回答；不得凭记忆编造气温或天气；从对话历史中推断用户所指城市并填入 location。
3. 纯闲聊或与法律/天气无关的泛知识：可直接回答，无需调用工具。
4. 若工具无结果，如实说明并建议用户换种问法或咨询专业人士。

判断提示：当前用户消息很短时，务必结合对话历史理解是否在续问天气或法律问题。
"""


def _history_to_langchain(messages: list[dict[str, Any]]) -> list[BaseMessage]:
    converted: list[BaseMessage] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "user":
            converted.append(HumanMessage(content=content))
        elif role == "assistant":
            converted.append(AIMessage(content=content))
        elif role == "system":
            converted.append(SystemMessage(content=content))
    return converted


def _extract_final_answer(messages: list[BaseMessage]) -> str:
    """从 Agent 返回的消息列表中提取最终回答。"""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = str(message.content or "").strip()
            if content and not message.tool_calls:
                return content
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = str(message.content or "").strip()
            if content:
                return content
    return ""


def _merge_run_config(llm_config: RunnableConfig) -> RunnableConfig:
    """合并 LangGraph 传入的 callbacks 与 Agent 步数上限。"""
    # 每轮工具调用 ≈ model + tools 两个图节点
    recursion_limit = settings.agent_max_tool_rounds * 2 + 3
    return merge_configs(llm_config, {"recursion_limit": recursion_limit})


async def run_tool_agent(
    *,
    deps: RuntimeDeps,
    llm: BaseChatModel,
    messages: list[dict[str, Any]],
    llm_config: RunnableConfig,
) -> dict[str, Any]:
    """运行 LangChain ``create_agent``，返回与 AgentState 兼容的字段更新。"""
    ctx = AgentToolContext()
    tools = build_agent_tools(deps, ctx)
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=_AGENT_SYSTEM,
        name="legal_assistant_agent",
    )

    result = await agent.ainvoke(
        {"messages": _history_to_langchain(messages)},  # type: ignore[arg-type]
        config=_merge_run_config(llm_config),
    )

    output_messages = result.get("messages", [])
    answer = _extract_final_answer(output_messages)
    if not answer:
        return {"error": "Agent 回复为空", "answer": None}

    tools_used = sorted(ctx.tools_used)
    citations = format_citations(ctx.retrieved_docs) if ctx.retrieved_docs else []
    if LEGAL_SEARCH_TOOL in ctx.tools_used:
        answer = append_disclaimer(answer)

    return {
        "tools_used": tools_used,
        "location": ctx.weather_result.location if ctx.weather_result else None,
        "retrieved_docs": ctx.retrieved_docs or None,
        "tool_result": ctx.weather_result,
        "answer": answer,
        "citations": citations or None,
        "error": None,
    }
