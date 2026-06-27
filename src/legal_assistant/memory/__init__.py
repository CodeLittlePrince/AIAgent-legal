"""会话记忆模块的公共入口。

本包提供对话历史的存储与读取能力，采用 Redis（热缓存）+ PostgreSQL（持久化）的双层架构。
外部代码通常只需导入 ``MemoryManager`` 即可使用完整功能。
"""

from legal_assistant.memory.database import Base, async_session_factory, engine, get_async_session
from legal_assistant.memory.manager import MemoryManager
from legal_assistant.memory.models import Message, Session
from legal_assistant.memory.postgres_store import PostgresStore
from legal_assistant.memory.redis_store import RedisStore

# 显式声明对外公开的符号，便于 ``from legal_assistant.memory import *`` 及 IDE 自动补全
__all__ = [
    "Base",
    "MemoryManager",
    "Message",
    "PostgresStore",
    "RedisStore",
    "Session",
    "async_session_factory",
    "engine",
    "get_async_session",
]
