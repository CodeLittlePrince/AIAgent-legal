"""Spec §11: remaining v1 non-goals — auth, multi-tenant, live crawl."""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from legal_assistant.main import create_app
from legal_assistant.memory.models import Message, Session
from legal_assistant.runtime.nodes import RuntimeDeps
from tests.helpers.mock_llm import make_tool_calling_mock_llm

ROOT = Path(__file__).resolve().parents[2]
DOCKER_COMPOSE = ROOT / "docker-compose.yml"
MAIN_PY = ROOT / "src" / "legal_assistant" / "main.py"

pytestmark = pytest.mark.non_goals

AUTH_PATH_KEYWORDS = ("login", "register", "signup", "signin", "oauth")


def _api_route_paths(app) -> list[str]:
    paths: list[str] = []
    for route in app.routes:
        path = getattr(route, "path", None)
        if path and path.startswith("/api/"):
            paths.append(path)
    return paths


def test_no_user_auth_api_routes():
    app = create_app(skip_db_init=True, skip_auto_ingest=True, skip_rag_warmup=True, mount_web_ui=False)
    api_paths = _api_route_paths(app)

    for path in api_paths:
        lowered = path.lower()
        for keyword in AUTH_PATH_KEYWORDS:
            assert keyword not in lowered, f"Auth route found (spec §11): {path}"


def test_no_multi_tenant_fields_in_models():
    session_columns = Session.__table__.columns.keys()
    message_columns = Message.__table__.columns.keys()

    for column in ("tenant_id", "org_id", "workspace_id", "user_id"):
        assert column not in session_columns, f"Multi-tenant field on Session: {column}"
        assert column not in message_columns, f"Multi-tenant field on Message: {column}"


def test_legal_docs_not_auto_crawled_on_startup():
    source = MAIN_PY.read_text(encoding="utf-8")
    assert "download_legal_docs" not in source
    assert "ingest_legal_documents" in source

    compose = DOCKER_COMPOSE.read_text(encoding="utf-8").lower()
    assert "download_legal_docs" not in compose
    assert "cron" not in compose
    assert "schedule" not in compose


def test_auto_ingest_only_indexes_local_profile():
    from legal_assistant import main as main_module

    source = inspect.getsource(main_module._maybe_auto_ingest)
    assert "download_legal_docs" not in source
    assert "ingest_legal_documents" in source


def test_sync_chat_still_returns_json():
    manager = AsyncMock()
    manager.load = AsyncMock(return_value=[])
    manager.save_turn = AsyncMock(return_value=None)
    llm = make_tool_calling_mock_llm()
    deps = RuntimeDeps(llm=llm, memory_manager=manager)
    app = create_app(
        memory_manager=manager,
        runtime_deps=deps,
        skip_db_init=True,
        skip_auto_ingest=True,
        mount_web_ui=False,
    )

    with TestClient(app) as client:
        response = client.post("/api/v1/chat", json={"message": "你好"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
