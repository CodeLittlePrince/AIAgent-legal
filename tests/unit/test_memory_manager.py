import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from legal_assistant.memory.manager import MemoryManager
from legal_assistant.memory.postgres_store import PostgresStore
from legal_assistant.memory.redis_store import RedisStore


@pytest.fixture
def session_id() -> str:
    return str(uuid.uuid4())


@pytest.fixture
def mock_redis() -> MagicMock:
    store = MagicMock(spec=RedisStore)
    store.get_messages = AsyncMock(return_value=None)
    store.set_messages = AsyncMock()
    return store


@pytest.fixture
def mock_postgres() -> MagicMock:
    store = MagicMock(spec=PostgresStore)
    store.create_session_if_missing = AsyncMock()
    store.append_message = AsyncMock()
    store.get_messages = AsyncMock(return_value=[])
    return store


@pytest.fixture
def manager(mock_redis: MagicMock, mock_postgres: MagicMock) -> MemoryManager:
    return MemoryManager(redis_store=mock_redis, postgres_store=mock_postgres)


@pytest.mark.asyncio
async def test_truncate_keeps_last_n_turns(manager: MemoryManager) -> None:
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3"},
    ]

    result = await manager.truncate(messages, 2)

    assert result == [
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3"},
    ]


@pytest.mark.asyncio
async def test_save_and_load_structure(
    session_id: str,
    manager: MemoryManager,
    mock_redis: MagicMock,
    mock_postgres: MagicMock,
) -> None:
    pg_messages = [
        {"role": "user", "content": "hello", "intent": "general"},
        {"role": "assistant", "content": "hi", "intent": "general"},
    ]
    mock_postgres.get_messages.return_value = pg_messages

    await manager.save_turn(session_id, "hello", "hi", "general")

    mock_postgres.create_session_if_missing.assert_called_once_with(session_id)
    assert mock_postgres.append_message.call_count == 2
    mock_postgres.append_message.assert_any_call(session_id, "user", "hello", "general")
    mock_postgres.append_message.assert_any_call(session_id, "assistant", "hi", "general")
    mock_redis.set_messages.assert_called_once_with(session_id, pg_messages)

    mock_redis.get_messages.return_value = None
    mock_postgres.get_messages.return_value = pg_messages

    loaded = await manager.load(session_id)

    assert loaded == pg_messages
    mock_redis.set_messages.assert_called_with(session_id, pg_messages)


@pytest.mark.asyncio
async def test_load_uses_redis_cache(
    session_id: str,
    manager: MemoryManager,
    mock_redis: MagicMock,
    mock_postgres: MagicMock,
) -> None:
    cached = [{"role": "user", "content": "cached"}]
    mock_redis.get_messages.return_value = cached

    result = await manager.load(session_id)

    assert result == cached
    mock_postgres.get_messages.assert_not_called()
    mock_redis.set_messages.assert_not_called()
