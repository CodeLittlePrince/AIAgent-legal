"""PostgreSQL 会话消息持久化存储层。"""

import uuid
from typing import Any

from sqlalchemy import delete, select, text

from legal_assistant.memory.database import async_session_factory, engine
from legal_assistant.memory.models import Message, Session


def _message_to_dict(message: Message) -> dict:
    result: dict[str, Any] = {
        "role": message.role,
        "content": message.content,
    }
    if message.metadata_ is not None:
        result["metadata"] = message.metadata_
    return result


class PostgresStore:
    async def create_session_if_missing(self, session_id: str) -> None:
        sid = uuid.UUID(session_id)
        async with async_session_factory() as db:
            existing = await db.get(Session, sid)
            if existing is None:
                db.add(Session(id=sid))
                await db.commit()

    async def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        sid = uuid.UUID(session_id)
        async with async_session_factory() as db:
            db.add(
                Message(
                    session_id=sid,
                    role=role,
                    content=content,
                    metadata_=metadata,
                )
            )
            await db.commit()

    async def get_messages(self, session_id: str) -> list[dict]:
        sid = uuid.UUID(session_id)
        async with async_session_factory() as db:
            result = await db.execute(
                select(Message)
                .where(Message.session_id == sid)
                .order_by(Message.created_at)
            )
            return [_message_to_dict(message) for message in result.scalars().all()]

    async def delete_session(self, session_id: str) -> bool:
        sid = uuid.UUID(session_id)
        async with async_session_factory() as db:
            existing = await db.get(Session, sid)
            if existing is None:
                return False
            await db.execute(delete(Message).where(Message.session_id == sid))
            await db.delete(existing)
            await db.commit()
            return True

    async def ping(self) -> bool:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        return True
