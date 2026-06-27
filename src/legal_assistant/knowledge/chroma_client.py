"""Chroma 向量数据库客户端工厂。

根据运行环境（内存、本地持久化、远程 HTTP）创建合适的 Chroma 客户端，
并提供集合的获取、创建与重置辅助函数。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import chromadb

from legal_assistant.config import settings

if TYPE_CHECKING:
    from chromadb import ClientAPI

# 法律知识库在 Chroma 中的默认集合名称
COLLECTION_NAME = "legal_knowledge"


def get_chroma_client(
    chroma_host: str | None = None,
    chroma_port: int | None = None,
    *,
    persist_dir: str | None = None,
    ephemeral: bool = False,
) -> ClientAPI:
    """创建并返回 Chroma 客户端实例。

    优先级（非 ephemeral 时）：
    1. 若指定 ``persist_dir`` 或配置中有 ``chroma_persist_dir`` → 本地持久化客户端
    2. 否则 → 连接远程 Chroma HTTP 服务

    Args:
        chroma_host: Chroma 服务主机名，默认读配置。
        chroma_port: Chroma 服务端口，默认读配置。
        persist_dir: 本地向量数据持久化目录。
        ephemeral: 为 True 时使用内存客户端（进程退出后数据丢失，适合测试）。

    Returns:
        已配置的 ``ClientAPI`` 实例。
    """
    if ephemeral:
        return chromadb.EphemeralClient()

    effective_persist_dir = persist_dir if persist_dir is not None else settings.chroma_persist_dir
    if effective_persist_dir:
        return chromadb.PersistentClient(path=effective_persist_dir)

    host = chroma_host if chroma_host is not None else settings.chroma_host
    port = chroma_port if chroma_port is not None else settings.chroma_port
    return chromadb.HttpClient(host=host, port=port)


def get_or_create_collection(client: ClientAPI, name: str = COLLECTION_NAME):
    """获取已有集合；不存在则创建新集合。

    使用 broad ``except`` 是因为 Chroma 在集合不存在时抛出的异常类型不固定。
    """
    try:
        return client.get_collection(name)
    except Exception:
        return client.create_collection(name)


def reset_collection(client: ClientAPI, name: str = COLLECTION_NAME):
    """删除并重建指定集合，用于全量重新入库前清空旧向量数据。

    删除失败（例如集合本不存在）时静默忽略，再创建新集合。
    """
    try:
        client.delete_collection(name)
    except Exception:
        pass
    return client.create_collection(name)
