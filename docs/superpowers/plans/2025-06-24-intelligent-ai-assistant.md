# 智能 AI 助手 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an industrial-grade REST API AI assistant with legal RAG, weather tools, general chat, and full planner/memory/knowledge/tool/runtime/evaluation/observability stack.

**Architecture:** LangGraph orchestrates intent routing across legal (LlamaIndex+Chroma RAG), weather (pluggable adapters), and general nodes. Memory uses Redis hot cache + PostgreSQL persistence. Langfuse traces every request; pytest covers unit/integration/RAG/LLM-judge/agent-benchmark layers.

**Tech Stack:** Python 3.11, FastAPI, LangGraph, LlamaIndex, Chroma, Redis, PostgreSQL, DeepSeek, Langfuse, Docker Compose, pytest

**Spec:** `docs/superpowers/specs/2025-06-24-intelligent-ai-assistant-design.md`

---

## File Map

| Path | Responsibility |
|------|----------------|
| `pyproject.toml` | Dependencies, pytest markers, package config |
| `docker-compose.yml` | api, postgres, redis, chroma, langfuse, langfuse-db |
| `src/legal_assistant/config.py` | pydantic-settings for all env vars |
| `src/legal_assistant/planner/` | Intent classification |
| `src/legal_assistant/memory/` | Redis + PostgreSQL session store |
| `src/legal_assistant/knowledge/` | Ingest + RAG retrieval |
| `src/legal_assistant/tools/weather/` | Open-Meteo + stub adapters |
| `src/legal_assistant/runtime/` | LangGraph StateGraph |
| `src/legal_assistant/observability/` | Langfuse + Prometheus |
| `src/legal_assistant/api/` | FastAPI routes |
| `scripts/download_legal_docs.py` | Fetch legal excerpts to profile/ |
| `tests/` | unit, integration, evaluation |

---

### Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `src/legal_assistant/__init__.py`
- Create: `src/legal_assistant/config.py`

- [ ] **Step 1: Write pyproject.toml**

```toml
[project]
name = "legal-assistant"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.111.0",
    "uvicorn[standard]>=0.30.0",
    "pydantic>=2.7.0",
    "pydantic-settings>=2.3.0",
    "langgraph>=0.2.0",
    "langchain-openai>=0.2.0",
    "langchain-core>=0.3.0",
    "llama-index>=0.11.0",
    "llama-index-vector-stores-chroma>=0.2.0",
    "chromadb>=0.5.0",
    "redis>=5.0.0",
    "sqlalchemy[asyncio]>=2.0.0",
    "asyncpg>=0.29.0",
    "alembic>=1.13.0",
    "httpx>=0.27.0",
    "langfuse>=2.0.0",
    "prometheus-client>=0.20.0",
    "sentence-transformers>=3.0.0",
    "pyyaml>=6.0.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0.0", "pytest-asyncio>=0.23.0", "pytest-httpx>=0.30.0"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
markers = [
    "slow: LLM/API calls",
    "benchmark: agent E2E benchmark",
    "integration: requires docker services",
]
testpaths = ["tests"]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: Write config.py**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/legal_assistant"
    redis_url: str = "redis://localhost:6379/0"

    chroma_host: str = "localhost"
    chroma_port: int = 8001

    embedding_provider: str = "local"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"

    weather_provider: str = "open_meteo"
    qweather_api_key: str = ""
    gaode_api_key: str = ""

    max_history_turns: int = 20
    redis_session_ttl_seconds: int = 86400

    langfuse_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    api_key: str = ""
    skip_auto_ingest: bool = False

    legal_disclaimer: str = "本回答仅供参考，不构成法律意见，具体问题请咨询执业律师。"

settings = Settings()
```

- [ ] **Step 3: Copy .env.example from spec §9**

- [ ] **Step 4: Install and verify**

Run: `cd SuperpowersTest && python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`
Expected: install succeeds

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .env.example .gitignore src/legal_assistant/
git commit -m "chore: scaffold project with config and dependencies"
```

---

### Task 2: Docker Compose Infrastructure

**Files:**
- Create: `docker-compose.yml`
- Create: `Dockerfile`
- Create: `scripts/wait-for-it.sh`

- [ ] **Step 1: Write docker-compose.yml** with services: `postgres`, `redis`, `chroma`, `langfuse-db`, `langfuse`, `api` per spec §7. Map ports 5432, 6379, 8001, 3000, 8000. Use named volumes for postgres, chroma, langfuse-db.

- [ ] **Step 2: Write Dockerfile**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml .
COPY src/ src/
COPY scripts/ scripts/
COPY alembic/ alembic/
COPY alembic.ini .
RUN pip install --no-cache-dir -e .
COPY profile/ profile/
EXPOSE 8000
CMD ["uvicorn", "legal_assistant.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Verify compose config**

Run: `docker compose config`
Expected: valid YAML, all 6 services listed

- [ ] **Step 4: Commit**

```bash
git add docker-compose.yml Dockerfile scripts/wait-for-it.sh
git commit -m "chore: add Docker Compose stack with Langfuse"
```

---

### Task 3: PostgreSQL Models + Alembic

**Files:**
- Create: `src/legal_assistant/memory/models.py`
- Create: `src/legal_assistant/memory/database.py`
- Create: `alembic.ini`
- Create: `alembic/env.py`
- Create: `alembic/versions/001_initial.py`
- Test: `tests/unit/test_memory_postgres.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_memory_postgres.py
import pytest
from legal_assistant.memory.models import Session, Message

def test_message_model_fields():
    assert "session_id" in Message.__table__.columns.keys()
    assert "role" in Message.__table__.columns.keys()
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `pytest tests/unit/test_memory_postgres.py -v`
Expected: FAIL module not found

- [ ] **Step 3: Implement models.py** with `Session` and `Message` tables per spec §3.2

- [ ] **Step 4: Implement database.py** with async engine + session factory

- [ ] **Step 5: Create Alembic migration 001_initial**

- [ ] **Step 6: Run test — expect PASS**

- [ ] **Step 7: Commit**

---

### Task 4: Redis + MemoryManager

**Files:**
- Create: `src/legal_assistant/memory/redis_store.py`
- Create: `src/legal_assistant/memory/postgres_store.py`
- Create: `src/legal_assistant/memory/manager.py`
- Test: `tests/unit/test_memory_manager.py`

- [ ] **Step 1: Write failing tests** for load/save roundtrip, Redis cache hit, PG fallback on Redis miss (mock Redis)

- [ ] **Step 2: Implement redis_store.py** — serialize messages as JSON list, TTL from settings

- [ ] **Step 3: Implement postgres_store.py** — create_session, append_message, get_messages (ordered by created_at)

- [ ] **Step 4: Implement manager.py**

```python
class MemoryManager:
    async def load(self, session_id: str) -> list[dict]: ...
    async def save_turn(self, session_id: str, user_msg: str, assistant_msg: str, intent: str | None) -> None: ...
    async def truncate(self, messages: list[dict], max_turns: int) -> list[dict]: ...
```

- [ ] **Step 5: Run tests — expect PASS**

Run: `pytest tests/unit/test_memory_manager.py -v`

- [ ] **Step 6: Commit**

---

### Task 5: Download Legal Documents

**Files:**
- Create: `scripts/download_legal_docs.py`
- Create: `profile/legal/README.md`
- Create: `profile/legal/*.md` (generated by script)

- [ ] **Step 1: Write download script** — fetch public legal excerpts (民法典合同编/侵权编、劳动法、劳动合同法、消费者权益保护法) from npc.gov.cn or fallback curated markdown content embedded in script if fetch fails. Save as Markdown with `#`/`##` headers.

- [ ] **Step 2: Write profile/legal/README.md** with disclaimer and source URLs

- [ ] **Step 3: Run script**

Run: `python scripts/download_legal_docs.py`
Expected: at least 4 `.md` files in `profile/legal/`

- [ ] **Step 4: Commit**

```bash
git add scripts/download_legal_docs.py profile/legal/
git commit -m "feat: add legal document download script and profile corpus"
```

---

### Task 6: Knowledge Ingest + Retriever

**Files:**
- Create: `src/legal_assistant/knowledge/ingest.py`
- Create: `src/legal_assistant/knowledge/retriever.py`
- Create: `scripts/ingest_knowledge.py`
- Test: `tests/unit/test_knowledge_retriever.py`

- [ ] **Step 1: Write failing test** — ingest local profile, query "试用期", expect top result source contains "劳动"

- [ ] **Step 2: Implement ingest.py** — LlamaIndex SimpleDirectoryReader, MarkdownNodeParser, ChromaVectorStore collection `legal_knowledge`, local embedding via HuggingFaceEmbedding

- [ ] **Step 3: Implement retriever.py** — `retrieve(query: str, top_k: int = 5) -> list[RetrievedDoc]` with score threshold 0.5

- [ ] **Step 4: Write ingest_knowledge.py CLI**

- [ ] **Step 5: Run test with Chroma** (mark `@pytest.mark.integration` if needed)

- [ ] **Step 6: Commit**

---

### Task 7: Weather Tool (Pluggable)

**Files:**
- Create: `src/legal_assistant/tools/base.py`
- Create: `src/legal_assistant/tools/registry.py`
- Create: `src/legal_assistant/tools/weather/open_meteo.py`
- Create: `src/legal_assistant/tools/weather/qweather.py` (stub raises NotImplementedError)
- Create: `src/legal_assistant/tools/weather/gaode.py` (stub)
- Test: `tests/unit/test_weather_tool.py`

- [ ] **Step 1: Write failing test**

```python
@pytest.mark.asyncio
async def test_open_meteo_beijing(httpx_mock):
    # mock geocoding + forecast responses
    adapter = OpenMeteoAdapter()
    result = await adapter.get_weather("北京")
    assert result.location
    assert result.temperature is not None
```

- [ ] **Step 2: Implement WeatherResult dataclass and WeatherAdapter Protocol in base.py**

- [ ] **Step 3: Implement OpenMeteoAdapter** — geocode via open-meteo geocoding API, forecast via forecast API

- [ ] **Step 4: Implement registry.py** — `get_weather_adapter(provider: str) -> WeatherAdapter`

- [ ] **Step 5: Run tests — expect PASS**

- [ ] **Step 6: Commit**

---

### Task 8: Planner

**Files:**
- Create: `src/legal_assistant/planner/intent.py`
- Create: `src/legal_assistant/planner/router.py`
- Test: `tests/unit/test_planner.py`

- [ ] **Step 1: Write failing tests**

```python
@pytest.mark.parametrize("msg,expected", [
    ("劳动合同试用期多久", "legal"),
    ("北京今天天气", "weather"),
    ("你好", "general"),
])
def test_rule_based_intent(msg, expected):
    assert classify_by_rules(msg) == expected
```

- [ ] **Step 2: Implement intent.py** — keyword rules for legal/weather/general

- [ ] **Step 3: Implement router.py** — `async def classify(message, history) -> PlanResult` with fields `intent`, `confidence`, `location`. LLM structured output via DeepSeek when rules ambiguous; mock LLM in unit tests.

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

---

### Task 9: LangGraph Runtime

**Files:**
- Create: `src/legal_assistant/runtime/state.py`
- Create: `src/legal_assistant/runtime/nodes.py`
- Create: `src/legal_assistant/runtime/graph.py`
- Create: `src/legal_assistant/knowledge/legal_qa.py`
- Test: `tests/unit/test_runtime_graph.py`

- [ ] **Step 1: Write AgentState TypedDict in state.py** per spec §3.5

- [ ] **Step 2: Implement nodes.py**
  - `legal_node`: retrieve → build prompt with citations → LLM → populate answer + citations
  - `weather_node`: extract/use location → tool → format LLM
  - `general_node`: direct LLM with history

- [ ] **Step 3: Implement graph.py** — StateGraph with conditional routing on intent

- [ ] **Step 4: Write failing integration-style unit test** with mocked LLM/tool/knowledge

- [ ] **Step 5: Run tests — expect PASS**

- [ ] **Step 6: Commit**

---

### Task 10: Observability (Langfuse + Metrics)

**Files:**
- Create: `src/legal_assistant/observability/langfuse_client.py`
- Create: `src/legal_assistant/observability/tracing.py`
- Create: `src/legal_assistant/observability/metrics.py`
- Test: `tests/unit/test_observability.py`

- [ ] **Step 1: Implement langfuse_client.py** — lazy init, no-op when `langfuse_enabled=False`

- [ ] **Step 2: Implement tracing.py** — context manager `trace_chat(session_id)` yielding trace_id; decorator `@span(name)` for planner/memory/knowledge/tool/llm

- [ ] **Step 3: Implement metrics.py** — Prometheus counters/histograms per spec §3.7

- [ ] **Step 4: Test no-op mode doesn't raise when Langfuse disabled

- [ ] **Step 5: Commit**

---

### Task 11: FastAPI Routes

**Files:**
- Create: `src/legal_assistant/api/schemas.py`
- Create: `src/legal_assistant/api/routes.py`
- Create: `src/legal_assistant/main.py`
- Test: `tests/integration/test_api_chat.py`

- [ ] **Step 1: Write schemas** — ChatRequest, ChatResponse, FeedbackRequest, Citation

- [ ] **Step 2: Write routes.py**
  - `POST /api/v1/chat` — orchestrate memory → planner → graph → save → response
  - `GET /api/v1/sessions/{id}`
  - `DELETE /api/v1/sessions/{id}`
  - `POST /api/v1/knowledge/reindex`
  - `POST /api/v1/feedback`
  - `GET /health`
  - `GET /metrics`

- [ ] **Step 3: Write main.py** — lifespan hook for DB/redis init, optional auto-ingest

- [ ] **Step 4: Write integration test** with TestClient + mocked graph

- [ ] **Step 5: Run tests**

Run: `pytest tests/integration/test_api_chat.py -v`

- [ ] **Step 6: Commit**

---

### Task 12: Evaluation — Unit + RAG

**Files:**
- Create: `src/legal_assistant/evaluation/golden_cases.yaml`
- Create: `tests/evaluation/test_rag.py`
- Expand: `tests/unit/` planner/memory/tool tests

- [ ] **Step 1: Write golden_cases.yaml** with ~20 legal questions and expected source file names

- [ ] **Step 2: Write test_rag.py** — compute recall@5 against golden cases

- [ ] **Step 3: Run fast evaluation suite**

Run: `pytest tests/evaluation/test_rag.py tests/unit/ -m "not slow" -v`

- [ ] **Step 4: Commit**

---

### Task 13: Evaluation — LLM Judge + Agent Benchmark

**Files:**
- Create: `src/legal_assistant/evaluation/agent_benchmark/tasks.yaml`
- Create: `src/legal_assistant/evaluation/agent_benchmark/runner.py`
- Create: `src/legal_assistant/evaluation/agent_benchmark/metrics.py`
- Create: `tests/evaluation/test_llm_judge.py`
- Create: `tests/evaluation/test_agent_benchmark.py`
- Create: `scripts/export_low_score_traces.py`

- [ ] **Step 1: Write tasks.yaml** — single-turn legal/weather/general, multi-turn weather followup, general→legal switch

- [ ] **Step 2: Implement runner.py** — execute tasks via httpx against running API, collect per-turn results

- [ ] **Step 3: Implement metrics.py** — Task Success, Intent Accuracy, Tool Success, Memory Coherence, Citation Compliance; output `agent_benchmark_report.json`

- [ ] **Step 4: Write test_llm_judge.py** (mark `@pytest.mark.slow`) — DeepSeek scores legal answers 1-5 on relevance, accuracy, disclaimer presence

- [ ] **Step 5: Write test_agent_benchmark.py** (mark `@pytest.mark.benchmark`)

- [ ] **Step 6: Write export_low_score_traces.py** — Langfuse API fetch scores<3 → append to tasks.yaml

- [ ] **Step 7: Commit**

---

### Task 14: End-to-End Docker Verification

**Files:**
- Modify: `docker-compose.yml` (healthchecks, depends_on conditions)
- Create: `README.md` (minimal — setup + curl examples only)

- [ ] **Step 1: Start stack**

Run: `docker compose up -d --build`
Expected: all services healthy within 120s

- [ ] **Step 2: Test legal chat**

```bash
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"劳动合同试用期最长多久？"}' | jq .
```
Expected: intent=legal, citations non-empty, disclaimer present

- [ ] **Step 3: Test weather chat**

```bash
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"上海今天天气怎么样？"}' | jq .
```
Expected: intent=weather, answer mentions temperature

- [ ] **Step 4: Test general chat**

```bash
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"你好，介绍一下你自己"}' | jq .
```
Expected: intent=general

- [ ] **Step 5: Verify Langfuse UI** at http://localhost:3000 shows trace spans

- [ ] **Step 6: Run benchmark**

Run: `pytest tests/evaluation/test_agent_benchmark.py -m benchmark`
Expected: report written, Task Success Rate > 0.8

- [ ] **Step 7: Commit**

```bash
git add README.md
git commit -m "feat: complete intelligent AI assistant MVP with evaluation and observability"
```

---

## Implementation Order Summary

```
Task 1 Scaffold → Task 2 Docker → Task 3 PG/Alembic → Task 4 Memory
→ Task 5 Legal docs → Task 6 Knowledge → Task 7 Tools → Task 8 Planner
→ Task 9 Runtime → Task 10 Observability → Task 11 API
→ Task 12-13 Evaluation → Task 14 E2E verify
```

## Self-Review Checklist

- [x] Spec §3.1–§3.7 each mapped to tasks 8,4,6,7,9,12-13,10
- [x] Agent benchmark + Langfuse covered in Tasks 10, 13, 14
- [x] profile/legal download in Task 5
- [x] No TBD placeholders
- [x] Docker + API + tests end-to-end in Task 14
