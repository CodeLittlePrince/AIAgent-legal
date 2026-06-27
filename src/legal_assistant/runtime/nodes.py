"""Agent 图各节点的实现与运行时依赖注入。

每个 ``make_*_node`` 工厂返回一个 async 函数，签名符合 LangGraph 节点约定：
接收 ``AgentState``，返回要合并进状态的部分字段字典。
依赖通过 ``RuntimeDeps`` 注入，便于单元测试替换 LLM、检索器等组件。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from legal_assistant.config import settings
from legal_assistant.knowledge.legal_qa import (
    append_disclaimer,
    build_legal_prompt,
    format_citations,
)
from legal_assistant.knowledge.retriever import LegalRetriever
from legal_assistant.memory.manager import MemoryManager
from legal_assistant.observability.langchain_tracing import langchain_invoke_config
from legal_assistant.observability.tracing import span
from legal_assistant.planner.router import LLMClassifier, classify
from legal_assistant.runtime.state import AgentNode, AgentState
from legal_assistant.tools.base import WeatherAdapter
from legal_assistant.tools.registry import get_weather_adapter


def _last_user_message(messages: list[dict[str, Any]]) -> str:
    """从消息列表中取最近一条用户消息的文本内容。

    从后向前遍历，忽略 assistant/system 消息；若无有效 user 消息则返回空字符串。

    Args:
        messages: 对话历史，每项含 ``role`` 与 ``content`` 字段。

    Returns:
        最后一条用户消息的 ``content`` 字符串，找不到时返回 ``""``。
    """
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
    return ""


def _history_to_langchain(messages: list[dict[str, Any]]) -> list[Any]:
    """将内部消息格式转换为 LangChain 消息对象列表。

    通用对话节点需要把完整历史交给 LLM；LangChain 使用
    ``HumanMessage`` / ``AIMessage`` / ``SystemMessage`` 区分角色。

    Args:
        messages: 内部格式的对话历史。

    Returns:
        LangChain 消息对象列表，供 ``llm.ainvoke`` 使用。
    """
    converted: list[Any] = []
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


def create_llm() -> ChatOpenAI:
    """根据应用配置创建默认的大语言模型客户端。

    使用 DeepSeek 兼容 OpenAI 的 API 格式；``trust_env=False`` 避免
    系统代理环境变量干扰 httpx 请求（与 Langfuse 等模块策略一致）。

    Returns:
        配置好的 ``ChatOpenAI`` 异步聊天模型实例。
    """
    return ChatOpenAI(
        model=settings.deepseek_model,
        api_key=SecretStr(settings.deepseek_api_key) if settings.deepseek_api_key else None,
        base_url=settings.deepseek_base_url,
        http_async_client=httpx.AsyncClient(trust_env=False),
    )


@dataclass
class RuntimeDeps:
    """Agent 运行时依赖的容器，支持懒加载与测试注入。

    各字段默认为 ``None``；首次通过 ``get_*`` 访问时才创建真实实例。
    测试时可传入 mock 对象，无需 patch 全局单例。

    Attributes:
        llm: 聊天模型，默认由 ``create_llm()`` 创建。
        retriever: 法律知识库检索器。
        memory_manager: 会话记忆持久化管理器。
        weather_adapter: 天气数据适配器。
        llm_classifier: 意图分类器（可选，传给 ``classify``）。
    """

    llm: BaseChatModel | None = None
    retriever: LegalRetriever | None = None
    memory_manager: MemoryManager | None = None
    weather_adapter: WeatherAdapter | None = None
    llm_classifier: LLMClassifier | None = None

    def get_llm(self) -> BaseChatModel:
        """获取 LLM 实例，未注入时使用默认 DeepSeek 客户端。"""
        return self.llm or create_llm()

    def get_retriever(self) -> LegalRetriever:
        """获取法律文档检索器实例。"""
        return self.retriever or LegalRetriever()

    def get_memory_manager(self) -> MemoryManager:
        """获取记忆管理器实例。"""
        return self.memory_manager or MemoryManager()

    def get_weather_adapter(self) -> WeatherAdapter:
        """获取天气适配器，按 ``settings.weather_provider`` 从 registry 解析。"""
        return self.weather_adapter or get_weather_adapter(settings.weather_provider)


def make_planner_node(deps: RuntimeDeps) -> AgentNode:
    """创建「规划/意图分类」节点。

    读取用户最新消息，调用 ``classify`` 判断意图（法律/天气/通用）及地点等信息。
    分类失败时降级为 ``general`` 意图，并将异常信息写入 ``error``。

    Args:
        deps: 运行时依赖，其中 ``llm_classifier`` 可覆盖默认分类器。

    Returns:
        符合 LangGraph 节点签名的 async 函数。
    """

    async def planner_node(state: AgentState) -> dict[str, Any]:
        message = _last_user_message(state.get("messages", []))
        history = state.get("messages", [])
        try:
            with span("planner.classify"):
                result = await classify(message, history, llm_classifier=deps.llm_classifier)
            return {
                "intent": result.intent,
                "location": result.location,
                "error": None,
            }
        except Exception as exc:
            # 分类失败不阻断流程，走通用分支并记录错误
            return {"intent": "general", "error": str(exc)}

    return planner_node


def make_legal_node(deps: RuntimeDeps) -> AgentNode:
    """创建「法律问答」节点。

    流程：检索相关法条/文档 → 拼装 prompt → LLM 生成回答 → 附加免责声明与引用。

    Args:
        deps: 需包含可用的 LLM 与 ``LegalRetriever``。

    Returns:
        更新 ``retrieved_docs``、``answer``、``citations`` 等字段的节点函数。
    """

    async def legal_node(state: AgentState) -> dict[str, Any]:
        message = _last_user_message(state.get("messages", []))
        llm_config = langchain_invoke_config()
        try:
            with span("retriever.search"):
                retriever = deps.get_retriever()
                docs = retriever.retrieve(message)
            prompt = build_legal_prompt(message, docs)
            with span("llm.legal_generate"):
                response = await deps.get_llm().ainvoke(
                    [HumanMessage(content=prompt)],
                    config=llm_config,
                )
            answer = append_disclaimer(str(response.content))
            citations = format_citations(docs) if docs else []
            return {
                "retrieved_docs": docs,
                "answer": answer,
                "citations": citations,
                "error": None,
            }
        except Exception as exc:
            return {"error": str(exc), "answer": None, "citations": []}

    return legal_node


def make_weather_node(deps: RuntimeDeps) -> AgentNode:
    """创建「天气查询」节点。

    先通过天气适配器拉取结构化数据，再让 LLM 根据数据用中文组织自然语言回复。

    Args:
        deps: 需包含 LLM 与 ``WeatherAdapter``。

    Returns:
        更新 ``tool_result`` 与 ``answer`` 的节点函数。
    """

    async def weather_node(state: AgentState) -> dict[str, Any]:
        message = _last_user_message(state.get("messages", []))
        # 优先使用 planner 解析出的 location，否则整句用户消息当作地点
        location = state.get("location") or message
        llm_config = langchain_invoke_config()
        try:
            with span("tool.weather"):
                weather = await deps.get_weather_adapter().get_weather(location)
            prompt = (
                "你是天气助手。请根据以下天气数据，用简洁中文回答用户。\n"
                f"地点: {weather.location}\n"
                f"温度: {weather.temperature}\n"
                f"天气: {weather.conditions}\n"
                f"预报: {weather.forecast_summary}\n"
                f"用户问题: {message}"
            )
            with span("llm.weather_summarize"):
                response = await deps.get_llm().ainvoke(
                    [HumanMessage(content=prompt)],
                    config=llm_config,
                )
            return {
                "tool_result": weather,
                "answer": str(response.content),
                "error": None,
            }
        except Exception as exc:
            return {"error": str(exc), "answer": None}

    return weather_node


def make_general_node(deps: RuntimeDeps) -> AgentNode:
    """创建「通用对话」节点。

    将完整对话历史转为 LangChain 消息后，直接交给 LLM 多轮对话，无检索或工具调用。

    Args:
        deps: 需包含可用的 LLM。

    Returns:
        更新 ``answer`` 字段的节点函数。
    """

    async def general_node(state: AgentState) -> dict[str, Any]:
        llm_config = langchain_invoke_config()
        try:
            history = _history_to_langchain(state.get("messages", []))
            with span("llm.general_chat"):
                response = await deps.get_llm().ainvoke(history, config=llm_config)
            return {"answer": str(response.content), "error": None}
        except Exception as exc:
            return {"error": str(exc), "answer": None}

    return general_node


def make_save_memory_node(deps: RuntimeDeps) -> AgentNode:
    """创建「保存记忆」节点。

    在回答生成后，将本轮 user/assistant 消息与 intent 写入持久化记忆。
    缺少 session_id、用户消息或 answer 时静默跳过（返回空 dict，不修改状态）。

    Args:
        deps: 需包含 ``MemoryManager``。

    Returns:
        通常返回 ``{}``；记忆写入失败时返回 ``{"error": "..."}``。
    """

    async def save_memory_node(state: AgentState) -> dict[str, Any]:
        session_id = state.get("session_id")
        user_message = _last_user_message(state.get("messages", []))
        answer = state.get("answer")
        intent = state.get("intent")

        if not session_id or not user_message or not answer:
            return {}

        try:
            with span("memory.save_turn"):
                await deps.get_memory_manager().save_turn(
                    session_id=session_id,
                    user_msg=user_message,
                    assistant_msg=answer,
                    intent=intent,
                )
        except Exception as exc:
            return {"error": str(exc)}
        return {}

    return save_memory_node
