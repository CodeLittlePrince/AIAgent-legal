"""PostgreSQL 会话消息持久化存储层。

负责会话与消息的 CRUD，是对话历史的权威数据源；Redis 缓存失效后可从此处回填。
"""

import uuid
from typing import Any

from sqlalchemy import delete, select, text

from legal_assistant.memory.database import async_session_factory, engine
from legal_assistant.memory.models import Message, Session


def _message_to_dict(message: Message) -> dict:
    """将 ORM ``Message`` 对象转换为与 Redis 缓存一致的字典格式。

    仅包含非空的可选字段（intent、metadata），便于上层统一处理。
    """
    result: dict[str, Any] = {
        "role": message.role,
        "content": message.content,
    }
    if message.intent is not None:
        result["intent"] = message.intent
    if message.metadata_ is not None:
        result["metadata"] = message.metadata_
    return result


class PostgresStore:
    """基于 PostgreSQL 的会话与消息持久化操作。"""

    async def create_session_if_missing(self, session_id: str) -> None:
        """若会话不存在则创建空会话记录。

        写入消息前调用，避免外键约束失败。
        """
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
        intent: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """向指定会话追加一条消息并提交事务。

        Args:
            session_id: 会话 UUID 字符串。
            role: 消息角色，如 ``user`` 或 ``assistant``。
            content: 消息正文。
            intent: 可选的意图标签。
            metadata: 可选的 JSON 扩展信息。
        """
        sid = uuid.UUID(session_id)
        async with async_session_factory() as db:
            db.add(
                Message(
                    session_id=sid,
                    role=role,
                    content=content,
                    intent=intent,
                    metadata_=metadata,
                )
            )
            await db.commit()

    async def get_messages(self, session_id: str) -> list[dict]:
        """按创建时间升序返回会话内全部消息。

        Returns:
            消息字典列表；会话无消息时返回空列表。
        """
        sid = uuid.UUID(session_id)
        async with async_session_factory() as db:
            result = await db.execute(
                select(Message)
                .where(Message.session_id == sid)
                .order_by(Message.created_at)  # 保证对话顺序与时间一致
            )
            return [_message_to_dict(message) for message in result.scalars().all()]

    async def delete_session(self, session_id: str) -> bool:
        """删除会话及其全部消息。

        Returns:
            会话存在且已删除返回 True；会话不存在返回 False。
        """
        sid = uuid.UUID(session_id)
        async with async_session_factory() as db:
            existing = await db.get(Session, sid)
            if existing is None:
                return False
            # 先删子表消息，再删会话记录
            await db.execute(delete(Message).where(Message.session_id == sid))
            await db.delete(existing)
            await db.commit()
            return True

    async def ping(self) -> bool:
        """健康检查：执行 ``SELECT 1`` 验证数据库连接。"""
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        return True
