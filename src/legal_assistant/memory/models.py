"""对话会话与消息的 SQLAlchemy ORM 模型定义。

一张 ``sessions`` 表对应一次用户对话，``messages`` 表存储该会话中的逐条消息。
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from legal_assistant.memory.database import Base


class Session(Base):
    """对话会话实体，对应数据库表 ``sessions``。

    每个会话由 UUID 唯一标识，可包含多条 ``Message`` 记录。
    """

    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),  # 记录更新时由数据库自动刷新时间戳
    )

    # 一对多关系：删除会话时需先处理关联消息（由 PostgresStore 显式删除）
    messages: Mapped[list["Message"]] = relationship(back_populates="session")


class Message(Base):
    """单条对话消息，对应数据库表 ``messages``。

    ``role`` 通常为 ``user`` 或 ``assistant``；``metadata_`` 为可选扩展字段。
    """

    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id"),  # 外键关联到所属会话
    )
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    # Python 属性名 metadata_ 避免与 SQLAlchemy 保留名冲突；数据库列名仍为 metadata
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    session: Mapped["Session"] = relationship(back_populates="messages")
