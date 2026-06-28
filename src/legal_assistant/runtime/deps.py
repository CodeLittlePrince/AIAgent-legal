"""Agent 运行时依赖注入容器。"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from legal_assistant.config import settings
from legal_assistant.knowledge.retriever import LegalRetriever
from legal_assistant.memory.manager import MemoryManager
from legal_assistant.tools.base import WeatherAdapter
from legal_assistant.tools.registry import get_weather_adapter


def create_llm() -> ChatOpenAI:
    """根据应用配置创建默认的大语言模型客户端。"""
    return ChatOpenAI(
        model=settings.deepseek_model,
        api_key=SecretStr(settings.deepseek_api_key) if settings.deepseek_api_key else None,
        base_url=settings.deepseek_base_url,
        http_async_client=httpx.AsyncClient(trust_env=False),
    )


@dataclass
class RuntimeDeps:
    """Agent 运行时依赖的容器，支持懒加载与测试注入。"""

    llm: BaseChatModel | None = None
    retriever: LegalRetriever | None = None
    memory_manager: MemoryManager | None = None
    weather_adapter: WeatherAdapter | None = None

    def get_llm(self) -> BaseChatModel:
        return self.llm or create_llm()

    def get_retriever(self) -> LegalRetriever:
        return self.retriever or LegalRetriever()

    def get_memory_manager(self) -> MemoryManager:
        return self.memory_manager or MemoryManager()

    def get_weather_adapter(self) -> WeatherAdapter:
        return self.weather_adapter or get_weather_adapter(settings.weather_provider)
