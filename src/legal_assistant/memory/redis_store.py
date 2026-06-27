"""Redis 会话消息热缓存层。

将会话消息序列化为 JSON 存入 Redis，加速频繁读取；带 TTL 自动过期，减轻内存压力。
"""

import json

import redis.asyncio as redis

from legal_assistant.config import settings


class RedisStore:
    """基于 Redis 的会话消息读写封装。

    键名格式为 ``legal_assistant:session:{session_id}:messages``，
    值为 JSON 数组，每个元素为一条消息字典（含 role、content 等字段）。
    """

    def __init__(self, client: redis.Redis | None = None) -> None:
        """初始化 Redis 客户端。

        Args:
            client: 可选的外部 Redis 客户端；为 None 时根据配置自动创建，
                并在 ``close()`` 时由本实例负责关闭连接。
        """
        self._client = client or redis.from_url(settings.redis_url, decode_responses=True)
        # 仅当客户端由本类创建时，close 才需要释放连接
        self._owns_client = client is None

    def _key(self, session_id: str) -> str:
        """生成指定会话在 Redis 中的键名。"""
        return f"legal_assistant:session:{session_id}:messages"

    async def get_messages(self, session_id: str) -> list[dict] | None:
        """读取会话消息列表。

        Returns:
            消息字典列表；键不存在或已过期时返回 None。
        """
        raw = await self._client.get(self._key(session_id))
        if raw is None:
            return None
        return json.loads(raw)

    async def set_messages(
        self,
        session_id: str,
        messages: list[dict],
        ttl: int | None = None,
    ) -> None:
        """写入或覆盖会话消息，并设置过期时间。

        Args:
            session_id: 会话 ID 字符串。
            messages: 消息字典列表。
            ttl: 过期秒数；为 None 时使用配置项 ``redis_session_ttl_seconds``。
        """
        effective_ttl = ttl if ttl is not None else settings.redis_session_ttl_seconds
        await self._client.set(
            self._key(session_id),
            json.dumps(messages),
            ex=effective_ttl,  # ex 表示以秒为单位的过期时间
        )

    async def delete_messages(self, session_id: str) -> None:
        """删除指定会话的缓存键（会话删除或需要强制刷新缓存时调用）。"""
        await self._client.delete(self._key(session_id))

    async def ping(self) -> bool:
        """健康检查：验证 Redis 连接是否可用。"""
        return bool(await self._client.ping())

    async def close(self) -> None:
        """关闭 Redis 连接（仅当客户端由本实例创建时执行）。"""
        if self._owns_client:
            await self._client.aclose()
