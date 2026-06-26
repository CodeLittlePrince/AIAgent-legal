from legal_assistant.config import settings
from legal_assistant.memory.postgres_store import PostgresStore
from legal_assistant.memory.redis_store import RedisStore


class MemoryManager:
    def __init__(
        self,
        redis_store: RedisStore | None = None,
        postgres_store: PostgresStore | None = None,
    ) -> None:
        self.redis = redis_store or RedisStore()
        self.postgres = postgres_store or PostgresStore()

    async def load(self, session_id: str) -> list[dict]:
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
        intent: str | None = None,
    ) -> None:
        await self.postgres.create_session_if_missing(session_id)
        await self.postgres.append_message(session_id, "user", user_msg, intent)
        await self.postgres.append_message(session_id, "assistant", assistant_msg, intent)

        messages = await self.postgres.get_messages(session_id)
        truncated = await self.truncate(messages, settings.max_history_turns)
        await self.redis.set_messages(session_id, truncated)

    async def truncate(self, messages: list[dict], max_turns: int) -> list[dict]:
        if max_turns <= 0:
            return []

        user_indices = [index for index, message in enumerate(messages) if message.get("role") == "user"]
        if len(user_indices) <= max_turns:
            return messages

        return messages[user_indices[-max_turns] :]

    async def delete_session(self, session_id: str) -> bool:
        deleted = await self.postgres.delete_session(session_id)
        await self.redis.delete_messages(session_id)
        return deleted
