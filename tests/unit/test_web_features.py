"""Web UI and SSE streaming features."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from legal_assistant.main import create_app
from legal_assistant.runtime.nodes import RuntimeDeps
from tests.helpers.mock_llm import make_tool_calling_mock_llm

WEB_DIST = Path(__file__).resolve().parents[2] / "web" / "dist"


@pytest.fixture
def client():
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
        mount_web_ui=True,
    )
    with TestClient(app) as test_client:
        yield test_client


def test_chat_stream_route_registered():
    app = create_app(skip_db_init=True, skip_auto_ingest=True, mount_web_ui=False)
    schema = app.openapi()
    assert "/api/v1/chat/stream" in schema["paths"]


def test_web_ui_index_when_dist_exists(client):
    if not WEB_DIST.exists():
        pytest.skip("web/dist not built — run `npm run build` in web/")

    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "法律智能助手" in response.text
