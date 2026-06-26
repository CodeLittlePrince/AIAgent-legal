from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from legal_assistant.api.routes import AppServices, router
from legal_assistant.config import settings
from legal_assistant.memory.database import Base, engine
from legal_assistant.memory.manager import MemoryManager
from legal_assistant.runtime.graph import build_agent_graph
from legal_assistant.runtime.nodes import RuntimeDeps


async def _init_database() -> None:
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


async def _maybe_auto_ingest() -> None:
    if settings.skip_auto_ingest:
        return

    try:
        from legal_assistant.knowledge.chroma_client import get_chroma_client
        from legal_assistant.knowledge.ingest import LEGAL_COLLECTION, ingest_legal_documents

        client = get_chroma_client()
        collection = client.get_or_create_collection(LEGAL_COLLECTION)
        if collection.count() == 0:
            ingest_legal_documents()
    except Exception:
        return


def create_app(
    *,
    memory_manager: MemoryManager | None = None,
    runtime_deps: RuntimeDeps | None = None,
    skip_db_init: bool = False,
    skip_auto_ingest: bool = False,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not skip_db_init:
            await _init_database()
        if not skip_auto_ingest:
            await _maybe_auto_ingest()

        manager = memory_manager or MemoryManager()
        deps = runtime_deps or RuntimeDeps(memory_manager=manager)
        if deps.memory_manager is None:
            deps.memory_manager = manager
        graph = build_agent_graph(deps)

        app.state.services = AppServices(
            memory_manager=manager,
            runtime_deps=deps,
            graph=graph,
        )
        yield

    app = FastAPI(title="Legal Assistant", lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()
