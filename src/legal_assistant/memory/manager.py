"""会话记忆统一管理器（Redis 缓存 + PostgreSQL 持久化）。

对外提供加载、保存一轮对话、截断历史、删除会话等高层 API，
隐藏底层双存储的读写与同步细节。
"""

from legal_assistant.config import settings
from legal_assistant.memory.postgres_store import PostgresStore
from legal_assistant.memory.redis_store import RedisStore


class MemoryManager:
    """协调 Redis 热缓存与 PostgreSQL 持久化的记忆管理器。

    读取策略：优先 Redis，未命中则从 PostgreSQL 加载并回填缓存。
    写入策略：先写 PostgreSQL，再按配置截断后更新 Redis。
    """

    def __init__(
        self,
        redis_store: RedisStore | None = None,
        postgres_store: PostgresStore | None = None,
    ) -> None:
        """初始化存储后端。

        Args:
            redis_store: 可选的 Redis 存储实例，便于测试注入 mock。
            postgres_store: 可选的 PostgreSQL 存储实例，便于测试注入 mock。
        """
        self.redis = redis_store or RedisStore()
        self.postgres = postgres_store or PostgresStore()

    async def load(self, session_id: str) -> list[dict]:
        """加载会话历史消息。

        缓存命中直接返回；否则从数据库读取，若有数据则写入 Redis 加速后续访问。

        Returns:
            消息字典列表，可能为空。
        """
        cached = await self.redis.get_messages(session_id)
        if cached is not None:
            return cached

        messages = await self.postgres.get_messages(session_id)
        if messages:
            await self.redis.set_messages(session_id, messages)
        return messages

    async def save_turn(
        self,
        session_id: str,
        user_msg: str,
        assistant_msg: str,
    ) -> None:
        """持久化一轮完整对话（用户消息 + 助手回复），并更新 Redis 缓存。"""
        await self.postgres.create_session_if_missing(session_id)
        await self.postgres.append_message(session_id, "user", user_msg)
        await self.postgres.append_message(session_id, "assistant", assistant_msg)

        messages = await self.postgres.get_messages(session_id)
        truncated = await self.truncate(messages, settings.max_history_turns)
        await self.redis.set_messages(session_id, truncated)

    async def truncate(self, messages: list[dict], max_turns: int) -> list[dict]:
        """按「轮次」截断消息列表，保留最近的若干轮用户发言及其后续消息。

        一轮以一条 ``role=user`` 的消息为起点；截断从第 N 轮用户消息的位置切片，
        从而保留该轮起的 user/assistant 完整交替。

        Args:
            messages: 按时间排序的完整消息列表。
            max_turns: 最多保留的用户轮次数；<=0 时返回空列表。

        Returns:
            截断后的消息子列表。
        """
        if max_turns <= 0:
            return []

        # 找出所有用户消息的索引，用于定位「第几轮」的起始位置
        user_indices = [index for index, message in enumerate(messages) if message.get("role") == "user"]
        if len(user_indices) <= max_turns:
            return messages

        # 从倒数第 max_turns 轮的用户消息处开始保留
        return messages[user_indices[-max_turns] :]

    async def delete_session(self, session_id: str) -> bool:
        """删除会话：同时清除 PostgreSQL 记录与 Redis 缓存。

        Returns:
            数据库中会话是否存在并已删除（与 ``PostgresStore.delete_session`` 一致）。
        """
        deleted = await self.postgres.delete_session(session_id)
        await self.redis.delete_messages(session_id)
        return deleted
