# Intelligent AI Assistant (Legal Assistant)

Industrial-grade REST API for conversational AI: **Chinese legal Q&A** (RAG), **weather queries**, and **general chat** via a unified **Tool Calling Agent** (LangGraph). LlamaIndex + Chroma retrieval, Redis/PostgreSQL memory, Langfuse observability.

## Prerequisites

- **Docker** and **Docker Compose** (v2+)
- **Python 3.11+** (for local development and tests)
- **`DEEPSEEK_API_KEY`** — required for LLM-powered chat, evaluation, and benchmarks ([DeepSeek API](https://platform.deepseek.com/))

Copy environment template and set your keys:

```bash
cp .env.example .env
# Edit .env — at minimum set DEEPSEEK_API_KEY
# For Langfuse tracing, also set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY
```

## Quick Start (Docker Compose)

Start the full stack (API, Postgres, Redis, Chroma, Langfuse):

```bash
docker compose up -d --build
```

Wait until services are healthy (first start may take 1–2 minutes while the embedding model downloads):

```bash
docker compose ps
curl -s http://localhost:8000/health | jq .
```

On first start, index legal documents (Docker sets `SKIP_AUTO_INGEST=true` for fast API boot):

```bash
curl -s -X POST http://localhost:8000/api/v1/knowledge/reindex | jq .
```

Services and ports:

| Service            | Port | Description                         |
|--------------------|------|-------------------------------------|
| api                | 8000 | FastAPI REST API                    |
| postgres           | 5432 | Session / message store             |
| redis              | 6379 | Hot session cache                   |
| chroma             | 8001 | Vector store (legal RAG)            |
| langfuse-web       | 3000 | Langfuse v3 UI (trace & evaluation) |
| langfuse-worker    | —    | Langfuse v3 background ingestion    |
| langfuse-clickhouse| —    | Langfuse analytics store            |
| langfuse-minio     | —    | Langfuse event/media object storage |

Stop the stack:

```bash
docker compose down
```

## API Examples

### Legal chat

```bash
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"劳动合同试用期最长多久？"}' | jq .
```

Expected: `tools_used` includes `search_legal_knowledge`, non-empty `citations`, `disclaimer` present.

### Weather chat

```bash
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"上海今天天气怎么样？"}' | jq .
```

Expected: `tools_used` includes `get_weather_forecast`, answer mentions temperature or conditions.

### General chat

```bash
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"你好，介绍一下你自己"}' | jq .
```

Expected: `tools_used` is empty (`[]`).

### Streaming chat (SSE)

```bash
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"message":"你好"}'
```

Events: `session` → `tools` → `delta` (answer chunks) → `citations` / `disclaimer` → `done`.

The synchronous JSON endpoint `POST /api/v1/chat` remains available for clients that do not use SSE.

### Health and metrics

```bash
curl -s http://localhost:8000/health | jq .
curl -s http://localhost:8000/metrics
```

## Running Tests

Install dev dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Unit tests

```bash
pytest tests/unit -v --tb=short
```

### Integration tests

```bash
pytest tests/integration -v --tb=short
```

### Evaluation tests

```bash
pytest tests/evaluation -v --tb=short
```

### Fast suite (CI — exclude slow LLM calls and benchmarks)

Includes spec §11 non-goals compliance (`tests/unit/test_non_goals.py`, marker `non_goals`):

```bash
pytest tests/unit tests/integration tests/evaluation -m "not slow and not benchmark" -v --tb=short
```

GitHub Actions runs the same command on push/PR (`.github/workflows/ci.yml`).

### Slow / benchmark suites (require `DEEPSEEK_API_KEY`)

```bash
pytest tests/evaluation -m slow -v --tb=short
pytest tests/evaluation/test_agent_benchmark.py -m benchmark -v --tb=short
```

## Web UI (React)

The chat interface lives in `web/` (Vite + React + TypeScript). It uses SSE streaming via `POST /api/v1/chat/stream`.

### Development (API + Vite dev server)

**1. Start infrastructure** (Postgres, Redis, Chroma — required for memory/RAG):

```bash
docker compose up -d postgres redis chroma
```

**2. Use `localhost` in `.env`** when running the API on your machine (not inside Docker):

```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/legal_assistant
REDIS_URL=redis://localhost:6379/0
CHROMA_HOST=localhost
CHROMA_PORT=8001
```

(`postgres` / `redis` / `chroma` hostnames only work inside the Docker network.)

**3. Start API** (with venv active or `uv run`):

```bash
uv run uvicorn legal_assistant.main:app --reload --port 8000
```

**4. Start frontend:**

```bash
cd web && npm install && npm run dev
```

Open **http://localhost:5173**

### Production / Docker

Build the frontend into `web/dist`, then FastAPI serves it at **http://localhost:8000/**:

```bash
cd web && npm install && npm run build
uvicorn legal_assistant.main:app --port 8000
```

`docker compose up --build` includes the React build in the API image.

## Langfuse UI

Open **http://localhost:3000** after starting the Langfuse v3 stack:

```bash
docker compose up -d langfuse-db langfuse-redis langfuse-clickhouse langfuse-minio langfuse-web langfuse-worker
```

The stack uses **Langfuse server v3** (`langfuse-web` + `langfuse-worker` + ClickHouse + MinIO), which matches the Python SDK v4 OTEL ingestion API.

**Upgrading from Langfuse v2:** v3 uses a different database schema. Reset the Langfuse Postgres volume once, then sign up again in the UI:

```bash
docker compose down
docker volume rm aiagent-legal_langfuse_db_data   # prefix may differ; check `docker volume ls`
docker compose up -d langfuse-db langfuse-redis langfuse-clickhouse langfuse-minio langfuse-web langfuse-worker
```

Create a project in Langfuse, copy the public/secret keys into `.env` (`LANGFUSE_ENABLED=true`, `LANGFUSE_HOST=http://localhost:3000`), and restart the API. Chat requests should appear as traces with session metadata.

## Architecture Modules

| Module | Path | Responsibility |
|--------|------|----------------|
| **API** | `src/legal_assistant/api/` | REST endpoints (`/chat`, sessions, health, metrics) |
| **Runtime / Agent** | `src/legal_assistant/runtime/`, `tools/builder.py` | LangGraph Tool Calling loop |
| **Memory** | `src/legal_assistant/memory/` | Redis hot cache + PostgreSQL persistence |
| **Knowledge** | `src/legal_assistant/knowledge/` | Legal doc ingest, Chroma RAG, retriever |
| **Tools** | `src/legal_assistant/tools/` | Pluggable weather adapters (Open-Meteo, QWeather, Gaode) |
| **Observability** | `src/legal_assistant/observability/` | Langfuse tracing, Prometheus metrics |
| **Evaluation** | `src/legal_assistant/evaluation/` | RAG metrics, LLM judge, agent benchmark |

## Non-Goals (v1)

Per [spec §11](docs/superpowers/specs/2025-06-24-intelligent-ai-assistant-design.md#11-非目标第一版不做), the first release still excludes:

| Excluded | v1 approach |
|----------|-------------|
| User registration / login / OAuth | No auth endpoints; optional `API_KEY` reserved for future middleware |
| Multi-tenant isolation | Sessions keyed by `session_id` only — no `tenant_id` / `user_id` |
| Real-time legal doc crawling | Manual or CI via `scripts/download_legal_docs.py` + `POST /api/v1/knowledge/reindex` |

**Now included:** React Web chat UI (`web/`, served at `/`) and SSE streaming (`POST /api/v1/chat/stream`). JSON chat (`POST /api/v1/chat`) remains for non-streaming clients.

Compliance for remaining non-goals: `tests/unit/test_non_goals.py`. Web/SSE features: `tests/unit/test_web_features.py`, `tests/integration/test_api_chat.py`.

## Project Layout

```
src/legal_assistant/   # Application package
web/                   # React chat UI (Vite)
profile/legal/         # Static legal Markdown documents
tests/unit/            # Unit tests
tests/integration/     # API integration tests
tests/evaluation/      # RAG, LLM judge, benchmark tests
scripts/               # Ingest, legal doc download, trace export
```
