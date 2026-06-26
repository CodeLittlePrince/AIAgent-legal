# Intelligent AI Assistant (Legal Assistant)

Industrial-grade REST API for multi-intent conversational AI: **Chinese legal Q&A** (RAG over static legal documents), **weather queries** (pluggable providers), and **general chat**. Built with LangGraph orchestration, LlamaIndex + Chroma retrieval, Redis/PostgreSQL memory, and Langfuse observability.

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

Services and ports:

| Service     | Port | Description              |
|-------------|------|--------------------------|
| api         | 8000 | FastAPI REST API         |
| postgres    | 5432 | Session / message store  |
| redis       | 6379 | Hot session cache        |
| chroma      | 8001 | Vector store (legal RAG) |
| langfuse    | 3000 | Trace & evaluation UI    |

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

Expected: `intent=legal`, non-empty `citations`, `disclaimer` present.

### Weather chat

```bash
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"上海今天天气怎么样？"}' | jq .
```

Expected: `intent=weather`, answer mentions temperature or conditions.

### General chat

```bash
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"你好，介绍一下你自己"}' | jq .
```

Expected: `intent=general`.

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

### Fast suite (exclude slow LLM calls and benchmarks)

```bash
pytest tests/unit tests/integration tests/evaluation -m "not slow and not benchmark" -v --tb=short
```

### Slow / benchmark suites (require `DEEPSEEK_API_KEY`)

```bash
pytest tests/evaluation -m slow -v --tb=short
pytest tests/evaluation/test_agent_benchmark.py -m benchmark -v --tb=short
```

## Langfuse UI

Open **http://localhost:3000** after `docker compose up`.

Create a project in Langfuse, copy the public/secret keys into `.env`, and restart the API. Chat requests will appear as trace spans with intent, tool, and retrieval metadata.

## Architecture Modules

| Module | Path | Responsibility |
|--------|------|----------------|
| **API** | `src/legal_assistant/api/` | REST endpoints (`/chat`, sessions, health, metrics) |
| **Planner** | `src/legal_assistant/planner/` | Intent classification and routing (legal / weather / general) |
| **Runtime** | `src/legal_assistant/runtime/` | LangGraph StateGraph agent execution |
| **Memory** | `src/legal_assistant/memory/` | Redis hot cache + PostgreSQL persistence |
| **Knowledge** | `src/legal_assistant/knowledge/` | Legal doc ingest, Chroma RAG, retriever |
| **Tools** | `src/legal_assistant/tools/` | Pluggable weather adapters (Open-Meteo, QWeather, Gaode) |
| **Observability** | `src/legal_assistant/observability/` | Langfuse tracing, Prometheus metrics |
| **Evaluation** | `src/legal_assistant/evaluation/` | RAG metrics, LLM judge, agent benchmark |

## Project Layout

```
src/legal_assistant/   # Application package
profile/legal/         # Static legal Markdown documents
tests/unit/            # Unit tests
tests/integration/     # API integration tests
tests/evaluation/      # RAG, LLM judge, benchmark tests
scripts/               # Ingest, legal doc download, trace export
```
