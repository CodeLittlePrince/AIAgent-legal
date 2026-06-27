"""Langfuse 客户端的单例管理与初始化。

Langfuse 用于 LLM 应用的可观测性（trace、span 等）。本模块负责：
- 按配置决定是否启用
- 懒加载单例客户端
- 初始化时临时屏蔽系统代理，避免 httpx 继承 SOCKS 代理导致连接失败
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

import httpx

from legal_assistant.config import settings

if TYPE_CHECKING:
    from langfuse import Langfuse

# 模块级单例：Langfuse 客户端实例与「是否已尝试初始化」标志
_client: Langfuse | None = None
_initialized = False

# 常见代理相关环境变量键名；Langfuse/httpx 可能从 shell 继承这些值
_PROXY_ENV_KEYS = (
    "ALL_PROXY",
    "all_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
)


@contextmanager
def _isolate_from_system_proxy() -> Iterator[None]:
    """临时从环境中移除代理变量，避免 Langfuse 的 httpx 客户端走 SOCKS 代理。

    Langfuse 在内部创建 httpx 客户端时会读取 ``trust_env=True`` 的默认行为，
    若 shell 配置了 SOCKS 代理可能导致异步上报失败。进入上下文时 pop 相关键，
    退出时在 finally 中恢复原有值。

    Yields:
        None
    """
    saved = {key: os.environ.pop(key) for key in _PROXY_ENV_KEYS if key in os.environ}
    try:
        yield
    finally:
        os.environ.update(saved)


def reset_langfuse_client() -> None:
    """重置 Langfuse 单例状态，主要用于测试。

    将 ``_client`` 置为 ``None`` 并将 ``_initialized`` 设为 ``False``，
    下次调用 ``get_langfuse()`` 会重新读取配置并创建客户端。
    """
    global _client, _initialized
    _client = None
    _initialized = False


def get_langfuse() -> Langfuse | None:
    """获取 Langfuse 客户端单例；未启用或缺少密钥时返回 ``None``。

    初始化逻辑（仅首次调用时执行）：
    1. 若 ``settings.langfuse_enabled`` 为 False → 直接返回 None
    2. 若缺少 public/secret key → 返回 None
    3. 在隔离代理的上下文中构造 ``Langfuse``，使用 ``trust_env=False`` 的同步 httpx 客户端

    Returns:
        已配置的 ``Langfuse`` 实例，或不可用时的 ``None``。
    """
    global _client, _initialized

    if not settings.langfuse_enabled:
        return None

    if _initialized:
        return _client

    _initialized = True
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return None

    from langfuse import Langfuse

    with _isolate_from_system_proxy():
        _client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            httpx_client=httpx.Client(timeout=30.0, trust_env=False),
        )
    return _client


def get_langfuse_client() -> Langfuse | None:
    """``get_langfuse`` 的别名，便于与旧代码或外部命名习惯兼容。"""
    return get_langfuse()
