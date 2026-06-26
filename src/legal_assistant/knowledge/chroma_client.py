from __future__ import annotations

from typing import TYPE_CHECKING

import chromadb

from legal_assistant.config import settings

if TYPE_CHECKING:
    from chromadb import ClientAPI

COLLECTION_NAME = "legal_knowledge"


def get_chroma_client(
    chroma_host: str | None = None,
    chroma_port: int | None = None,
    *,
    persist_dir: str | None = None,
    ephemeral: bool = False,
) -> ClientAPI:
    if ephemeral:
        return chromadb.EphemeralClient()

    effective_persist_dir = persist_dir if persist_dir is not None else settings.chroma_persist_dir
    if effective_persist_dir:
        return chromadb.PersistentClient(path=effective_persist_dir)

    host = chroma_host if chroma_host is not None else settings.chroma_host
    port = chroma_port if chroma_port is not None else settings.chroma_port
    return chromadb.HttpClient(host=host, port=port)


def get_or_create_collection(client: ClientAPI, name: str = COLLECTION_NAME):
    try:
        return client.get_collection(name)
    except Exception:
        return client.create_collection(name)


def reset_collection(client: ClientAPI, name: str = COLLECTION_NAME):
    try:
        client.delete_collection(name)
    except Exception:
        pass
    return client.create_collection(name)
