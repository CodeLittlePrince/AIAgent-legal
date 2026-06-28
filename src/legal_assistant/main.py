"""FastAPI 应用入口模块。

负责创建 Web 服务、初始化数据库与知识库、挂载 API 路由和前端静态页面。
运行 `uvicorn legal_assistant.main:app` 即可启动服务。
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from legal_assistant.api.routes import AppServices, router
from legal_assistant.config import settings
from legal_assistant.memory.database import Base, engine
from legal_assistant.memory.manager import MemoryManager
from legal_assistant.runtime.deps import RuntimeDeps
from legal_assistant.runtime.graph import build_agent_graph

# 前端构建产物目录：main.py 位于 src/legal_assistant/，向上两级到项目根，再进入 web/dist
WEB_DIST = Path(__file__).resolve().parents[2] / "web" / "dist"


async def _init_database() -> None:
    """在应用启动时创建数据库表（若尚不存在）。

    使用 SQLAlchemy 的 metadata.create_all，不会删除已有数据。
    """
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


async def _maybe_warmup_rag(deps: RuntimeDeps) -> None:
    """启动时预加载 embedding + rerank 模型，避免首次检索才下载 HuggingFace 权重。"""
    if not settings.rag_warmup_on_startup:
        return

    try:
        import asyncio

        from legal_assistant.knowledge.warmup import warmup_rag_models

        deps.retriever = await asyncio.to_thread(warmup_rag_models)
    except Exception:
        # Chroma 未就绪或网络异常时不阻断服务启动
        return


async def _maybe_auto_ingest() -> None:
    """若知识库为空，则自动导入法律文档。

    可通过 settings.skip_auto_ingest 跳过此步骤。
    任何异常都会被静默忽略，避免启动失败（例如 Chroma 尚未就绪）。
    """
    if settings.skip_auto_ingest:
        return

    try:
        from legal_assistant.knowledge.chroma_client import get_chroma_client
        from legal_assistant.knowledge.ingest import LEGAL_COLLECTION, ingest_legal_documents

        client = get_chroma_client()
        collection = client.get_or_create_collection(LEGAL_COLLECTION)
        # 仅在集合中没有任何文档时才执行入库，避免重复导入
        if collection.count() == 0:
            ingest_legal_documents()
    except Exception:
        return


def _mount_web_ui(app: FastAPI) -> None:
    """挂载前端单页应用（SPA）的静态资源。

    若 web/dist 目录不存在（例如未执行 npm build），则跳过挂载。

    Args:
        app: FastAPI 应用实例。
    """
    if not WEB_DIST.exists():
        return

    assets_dir = WEB_DIST / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="web-assets")

    @app.get("/")
    async def web_index() -> FileResponse:
        """返回 SPA 首页。"""
        return FileResponse(WEB_DIST / "index.html")

    @app.get("/{full_path:path}")
    async def web_spa_fallback(full_path: str) -> FileResponse:
        """SPA 路由回退：非 API 路径优先返回静态文件，否则回退到 index.html。

        这样前端使用 React Router 等客户端路由时，刷新页面仍能正确加载。
        """
        # 以 api/ 开头的路径交给 FastAPI 路由处理，此处不拦截
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        file_path = WEB_DIST / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        # 未知路径统一返回 index.html，由前端路由接管
        return FileResponse(WEB_DIST / "index.html")


def create_app(
    *,
    memory_manager: MemoryManager | None = None,
    runtime_deps: RuntimeDeps | None = None,
    skip_db_init: bool = False,
    skip_auto_ingest: bool = False,
    skip_rag_warmup: bool = False,
    mount_web_ui: bool = True,
) -> FastAPI:
    """创建并配置 FastAPI 应用实例。

    支持注入依赖以便单元测试替换真实组件。

    Args:
        memory_manager: 自定义会话记忆管理器；默认在启动时新建。
        runtime_deps: Agent 运行时依赖（LLM、工具等）；默认基于 memory_manager 构建。
        skip_db_init: 为 True 时跳过数据库表初始化（测试常用）。
        skip_auto_ingest: 为 True 时跳过自动法律文档入库。
        skip_rag_warmup: 为 True 时跳过 RAG 模型预热（测试常用）。
        mount_web_ui: 为 True 时挂载 web/dist 静态前端。

    Returns:
        配置完成的 FastAPI 应用。
    """
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """应用生命周期：启动时初始化资源，关闭时清理（当前仅 yield，无额外清理）。"""
        if not skip_db_init:
            await _init_database()
        if not skip_auto_ingest:
            await _maybe_auto_ingest()

        manager = memory_manager or MemoryManager()
        deps = runtime_deps or RuntimeDeps(memory_manager=manager)
        if deps.memory_manager is None:
            deps.memory_manager = manager
        if not skip_rag_warmup and runtime_deps is None:
            await _maybe_warmup_rag(deps)
        # 构建 LangGraph 状态图，作为对话 Agent 的执行引擎
        graph = build_agent_graph(deps)

        # 将核心服务挂到 app.state，供路由层通过 get_services 获取
        app.state.services = AppServices(
            memory_manager=manager,
            runtime_deps=deps,
            graph=graph,
        )
        yield

    app = FastAPI(title="Legal Assistant", lifespan=lifespan)
    # CORS：允许浏览器跨域访问（开发阶段允许所有来源）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    if mount_web_ui:
        _mount_web_ui(app)
    return app


# uvicorn 默认加载的应用对象
app = create_app()
