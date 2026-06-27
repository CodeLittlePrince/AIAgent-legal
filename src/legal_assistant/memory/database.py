"""PostgreSQL 异步数据库连接配置。

使用 SQLAlchemy 2.0 的异步引擎与会话工厂，供 ``PostgresStore`` 及 ORM 模型使用。
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from legal_assistant.config import settings


class Base(DeclarativeBase):
    """所有 ORM 模型的声明式基类。

    ``Session`` 与 ``Message`` 均继承此类，以便 SQLAlchemy 统一管理表结构映射。
    """

    pass


# 全局异步引擎：连接字符串来自配置，echo=False 表示不在控制台打印 SQL
engine = create_async_engine(settings.database_url, echo=False)

# 会话工厂：每次调用可创建独立的 AsyncSession；expire_on_commit=False 避免提交后对象属性过期
async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 依赖注入用的异步会话生成器。

    用法示例::

        @app.get("/example")
        async def example(db: AsyncSession = Depends(get_async_session)):
            ...

    请求结束时上下文管理器会自动关闭会话。
    """
    async with async_session_factory() as session:
        yield session
