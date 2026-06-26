import json

import redis.asyncio as redis

from legal_assistant.config import settings


class RedisStore:
    def __init__(self, client: redis.Redis | None = None) -> None:
        self._client = client or redis.from_url(settings.redis_url, decode_responses=True)
        self._owns_client = client is None

    def _key(self, session_id: str) -> str:
        return f"legal_assistant:session:{session_id}:messages"

    async def get_messages(self, session_id: str) -> list[dict] | None:
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
        effective_ttl = ttl if ttl is not None else settings.redis_session_ttl_seconds
        await self._client.set(
            self._key(session_id),
            json.dumps(messages),
            ex=effective_ttl,
        )

    async def delete_messages(self, session_id: str) -> None:
        await self._client.delete(self._key(session_id))

    async def ping(self) -> bool:
        return bool(await self._client.ping())

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()
