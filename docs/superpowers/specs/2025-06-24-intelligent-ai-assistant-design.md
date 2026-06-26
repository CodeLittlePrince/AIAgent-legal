# 智能 AI 助手 — 系统设计规格

**日期：** 2025-06-24  
**状态：** 待实现  
**方案：** LangGraph + LlamaIndex 混合架构（方案 1）

---

## 1. 概述

构建一个工业级智能 AI 助手 REST API 服务，具备以下能力：

- **中国法律问答**：基于 `profile/legal/` 静态法律文档 + Chroma RAG
- **天气查询**：可插拔 Tool，默认 Open-Meteo，预留和风/高德适配器
- **通用问答**：开放域对话

Agent 架构包含六大模块：**Planner、Memory、Knowledge、Tool、Runtime、Evaluation**，并补充 **Observability**（Langfuse 类运行时质量观测）。

### 已确认技术选型

| 类别 | 选择 |
|------|------|
| 语言/框架 | Python 3.11+、FastAPI、Uvicorn |
| LLM | DeepSeek（OpenAI 兼容接口） |
| 编排 | LangGraph StateGraph |
| RAG | LlamaIndex + Chroma |
| Memory | Redis（热缓存）+ PostgreSQL（持久化）+ Chroma（向量，法律知识库 + 可选语义记忆） |
| 部署 | Docker Compose |
| 天气 Tool | 可插拔；默认 OpenMeteoAdapter |
| Evaluation | 单元 + 集成 + RAG + LLM Judge + Agent E2E Benchmark |
| Observability | Langfuse 自托管 + Prometheus 指标（可选） |

---

## 2. 项目目录结构

```
SuperpowersTest/
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── .env.example
├── profile/
│   └── legal/
│       ├── README.md                 # 免责声明与数据来源说明
│       ├── 民法典_节选.md
│       ├── 劳动法_节选.md
│       └── ...
├── src/
│   └── legal_assistant/
│       ├── main.py
│       ├── config.py
│       ├── api/
│       │   ├── routes.py
│       │   └── schemas.py
│       ├── planner/
│       │   ├── intent.py
│       │   └── router.py
│       ├── runtime/
│       │   ├── graph.py
│       │   ├── nodes.py
│       │   └── state.py
│       ├── memory/
│       │   ├── redis_store.py
│       │   ├── postgres_store.py
│       │   └── manager.py
│       ├── knowledge/
│       │   ├── ingest.py
│       │   ├── retriever.py
│       │   └── legal_qa.py
│       ├── tools/
│       │   ├── base.py
│       │   ├── registry.py
│       │   └── weather/
│       │       ├── open_meteo.py
│       │       ├── qweather.py
│       │       └── gaode.py
│       ├── observability/
│       │   ├── langfuse_client.py
│       │   ├── tracing.py
│       │   └── metrics.py
│       └── evaluation/
│           ├── golden_cases.yaml
│           ├── agent_benchmark/
│           │   ├── tasks.yaml
│           │   ├── runner.py
│           │   └── metrics.py
│           ├── test_unit.py
│           ├── test_integration.py
│           ├── test_rag.py
│           ├── test_llm_judge.py
│           └── test_agent_benchmark.py
├── scripts/
│   ├── download_legal_docs.py
│   ├── ingest_knowledge.py
│   └── export_low_score_traces.py    # 线上低分 trace → benchmark 用例
├── tests/
│   ├── unit/
│   ├── integration/
│   └── evaluation/
└── docs/superpowers/specs/
    └── 2025-06-24-intelligent-ai-assistant-design.md
```

---

## 3. 六模块 + Observability 职责

### 3.1 Planner

- 分析用户消息 + 会话历史，输出结构化 intent
- Intent 类型：`legal` | `weather` | `general`
- 实现：规则 fast-path（关键词）+ DeepSeek structured JSON 二次确认
- 对 weather intent 同时提取 `location`（structured extraction）
- 复杂问题可输出子步骤计划（第一版仅单步路由，子步骤为扩展点）

### 3.2 Memory

三层存储，统一由 `MemoryManager` 暴露接口：

| 层 | 技术 | 用途 |
|----|------|------|
| 热缓存 | Redis（TTL 24h） | 当前会话最近 N 轮消息 |
| 持久化 | PostgreSQL | 全量会话与消息记录 |
| 语义（可选 v1.1） | Chroma collection `user_memory` | 跨会话长期事实记忆 |

**读取路径：** Redis hit → 返回；miss → PostgreSQL → 回填 Redis  
**写入路径：** 每轮对话 → Redis → PostgreSQL（同步或异步均可，第一版同步）

**PostgreSQL 表：**

```sql
-- sessions
id          UUID PRIMARY KEY
created_at  TIMESTAMPTZ
updated_at  TIMESTAMPTZ

-- messages
id          UUID PRIMARY KEY
session_id  UUID REFERENCES sessions(id)
role        VARCHAR(16)   -- user | assistant | system
content     TEXT
intent      VARCHAR(16)   -- nullable
metadata    JSONB
created_at  TIMESTAMPTZ
```

上下文截断：送 LLM 前保留最近 **20 轮**（可配置 `MAX_HISTORY_TURNS`）。

### 3.3 Knowledge

- 源文件：`profile/legal/*.md`（脚本自动下载公开法律节选）
- Ingest：LlamaIndex 按 Markdown 标题 chunk（512 token，overlap 64）
- Embedding：DeepSeek embedding API，或本地 `BAAI/bge-small-zh-v1.5`（通过 `EMBEDDING_PROVIDER` 切换）
- 向量库：Chroma collection `legal_knowledge`
- 回答约束：
  - 必须基于检索片段
  - 返回 `citations`（source + excerpt）
  - 固定附加免责声明
  - 检索置信度低于阈值时，明确提示「未找到可靠法条，建议咨询执业律师」

### 3.4 Tool

可插拔天气 Tool：

```python
class WeatherAdapter(Protocol):
    async def get_weather(self, location: str) -> WeatherResult: ...
```

| 适配器 | 环境变量 `WEATHER_PROVIDER` | 说明 |
|--------|----------------------------|------|
| OpenMeteoAdapter | `open_meteo`（默认） | 无需 API Key |
| QWeatherAdapter | `qweather` | 预留，需 `QWEATHER_API_KEY` |
| GaodeAdapter | `gaode` | 预留，需 `GAODE_API_KEY` |

`WeatherResult` 包含：location、temperature、conditions、forecast_summary、raw_source。

### 3.5 Runtime

LangGraph `StateGraph` 编排：

```
START → load_memory → planner → route
                                  ├─ legal_node  → (RAG + LLM)
                                  ├─ weather_node → (Tool + LLM)
                                  └─ general_node → (LLM)
                              → save_memory → END
```

**AgentState 字段：**

```python
session_id: str
messages: list[Message]
intent: str | None
location: str | None
retrieved_docs: list[Document] | None
tool_result: WeatherResult | None
answer: str | None
citations: list[Citation] | None
error: str | None
```

DeepSeek 调用通过 `langchain-openai` 配置 `base_url=https://api.deepseek.com`。

### 3.6 Evaluation

七层评测体系：

| 层级 | 目录 | 内容 |
|------|------|------|
| 1. 单元 | `tests/unit/` | Planner 路由、Memory CRUD、Tool mock |
| 2. 集成 | `tests/integration/` | `/chat` E2E（docker-compose test profile） |
| 3. RAG | `tests/evaluation/test_rag.py` | recall@k、命中源文件（约 20 条 golden 法律问题） |
| 4. LLM Judge | `tests/evaluation/test_llm_judge.py` | 法律回答相关性/准确性/免责声明（1-5 分） |
| 5. Agent Benchmark | `tests/evaluation/test_agent_benchmark.py` | **Agent 整体 E2E 多轮评测** |
| 6. Observability | 运行时 Langfuse trace | 见 §3.7 |
| 7. Feedback Loop | `scripts/export_low_score_traces.py` | 线上低分 trace 导出为新 benchmark 用例 |

#### Agent Benchmark 规格

**任务集** `evaluation/agent_benchmark/tasks.yaml` 覆盖：

- 单轮 legal / weather / general
- 多轮追问（如「北京天气」→「那明天呢？」，测 memory + intent）
- 跨 intent 切换（general → legal）
- 错误输入与降级场景

**Agent 级指标：**

| 指标 | 说明 |
|------|------|
| Task Success Rate | 整条对话是否完成预期任务 |
| Intent Accuracy | 每轮 Planner 路由是否正确 |
| Tool Success Rate | 该调 Tool 时是否调对且成功 |
| Memory Coherence | 追问是否正确引用前文 |
| Citation Compliance | legal 是否带 citations + disclaimer |
| Latency P95 | 端到端响应时间 |
| Cost per Task | DeepSeek token 消耗 |

**输出：** `agent_benchmark_report.json`（总分、分 intent 得分、与上次 diff）

**运行命令：**

```bash
pytest tests/evaluation/test_agent_benchmark.py -m benchmark
pytest tests/evaluation/ -m "not slow"          # CI 快速集
pytest tests/evaluation/test_llm_judge.py -m slow # 含 LLM Judge
```

### 3.7 Observability（Langfuse 类运行时观测）

与离线 Evaluation 互补：Evaluation 用于 CI/回归；Observability 用于每次真实请求的质量追踪与排障。

**Docker Compose 服务：**

```yaml
langfuse:       # Web UI，默认 port 3000
langfuse-db:    # Langfuse 专用 PostgreSQL（与业务 PG 分离）
```

**环境变量：**

```
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=http://langfuse:3000
```

本地开发可设 `LANGFUSE_ENABLED=false` 跳过上报。

**每个 `/chat` 请求一条 Trace，Span 规范：**

| Span 名称 | 记录字段 |
|-----------|----------|
| `planner` | intent, confidence, latency_ms |
| `memory.load` | cache_hit, message_count |
| `memory.save` | message_count |
| `knowledge.retrieve` | query, top_k, scores, doc_ids |
| `tool.weather` | provider, location, success, latency_ms |
| `llm.generate` | model, input_tokens, output_tokens, latency_ms, cost |
| `graph.total` | intent, total_latency_ms, success |

**Prometheus 指标（`observability/metrics.py`）：**

- `chat_requests_total{intent, status}`
- `chat_latency_seconds`（histogram）
- `llm_tokens_total{model, direction}`
- `tool_calls_total{tool, status}`

**Feedback API：**

```
POST /api/v1/feedback
{
  "trace_id": "...",
  "score": 1,           // 1=positive, 0=negative
  "comment": "optional"
}
```

写入 Langfuse Scores API，供 `export_low_score_traces.py` 导出为 benchmark 用例。

---

## 4. API 设计

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/chat` | 主对话接口 |
| GET | `/api/v1/sessions/{id}` | 获取会话历史 |
| DELETE | `/api/v1/sessions/{id}` | 清除会话 |
| POST | `/api/v1/knowledge/reindex` | 重建法律向量索引 |
| POST | `/api/v1/feedback` | 用户反馈 → Langfuse score |
| GET | `/health` | 健康检查（Redis/PG/Chroma/Langfuse 连通性） |
| GET | `/metrics` | Prometheus metrics（可选） |

### ChatRequest

```json
{
  "session_id": "uuid-or-null",
  "message": "劳动合同试用期最长多久？"
}
```

`session_id` 为 null 时服务端生成新 UUID。

### ChatResponse

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "intent": "legal",
  "answer": "...",
  "citations": [
    {"source": "劳动法_节选.md", "excerpt": "..."}
  ],
  "disclaimer": "本回答仅供参考，不构成法律意见，具体问题请咨询执业律师。",
  "trace_id": "langfuse-trace-id"
}
```

第一版 **无用户认证**；`.env` 预留 `API_KEY`，后续可通过 middleware 启用。

---

## 5. 核心数据流

```
Client
  │ POST /api/v1/chat
  ▼
FastAPI
  │ ① Langfuse: start trace
  │ ② MemoryManager.load(session_id)     [Redis → PG]
  │ ③ Planner.classify(message, history)
  ▼
LangGraph Runtime
  │ route by intent
  ├─ legal:  Knowledge.retrieve → LLM(generate + citations)
  ├─ weather: Tool.get_weather → LLM(format)
  └─ general: LLM(direct)
  │ ④ MemoryManager.save(session_id, turn)
  │ ⑤ Langfuse: end trace
  ▼
ChatResponse
```

---

## 6. 法律文档（profile/legal/）

**下载脚本：** `scripts/download_legal_docs.py`

从公开来源获取法律文档**节选**（非全文替代正式法律文本）：

- 民法典（合同编、侵权责任编相关章节）
- 劳动法
- 劳动合同法
- 消费者权益保护法

格式：Markdown，带 `#` / `##` 标题层级便于 chunk。  
`profile/legal/README.md` 说明数据来源、更新时间、免责声明。

**Ingest 脚本：** `scripts/ingest_knowledge.py`  
读取 `profile/legal/` → chunk → embed → Chroma `legal_knowledge`。

---

## 7. Docker Compose

```yaml
services:
  api:           # FastAPI, port 8000
  postgres:      # 业务 DB, port 5432, volume
  redis:         # port 6379
  chroma:        # port 8001, volume
  langfuse:      # port 3000
  langfuse-db:   # Langfuse 专用 PG
```

**API 启动流程：**

1. 等待 postgres / redis / chroma 就绪
2. Alembic migrate
3. 若 Chroma collection 为空，自动 ingest（可通过 `SKIP_AUTO_INGEST=true` 跳过）
4. uvicorn 启动

---

## 8. 错误处理

| 场景 | HTTP | 行为 |
|------|------|------|
| DeepSeek API 失败 | 503 | 重试 2 次后返回友好提示 |
| Chroma 不可用 | 200 | legal intent 降级，告知知识库暂不可用 |
| 天气 API 失败 | 200 | 返回「无法获取天气，请确认城市名」 |
| Redis 不可用 | 200 | 降级直读 PostgreSQL |
| PostgreSQL 不可用 | 503 | 无法持久化，拒绝服务 |
| 超长历史 | 200 | 截断最近 N 轮 |
| 未知 intent | 200 | fallback → general |
| Langfuse 不可用 | 200 | 跳过 trace 上报，不影响主流程 |

所有错误写入 structured log；Langfuse span 标记 `error=true`。

---

## 9. 环境变量（.env.example）

```bash
# LLM
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@postgres:5432/legal_assistant
REDIS_URL=redis://redis:6379/0

# Chroma
CHROMA_HOST=chroma
CHROMA_PORT=8001

# Embedding
EMBEDDING_PROVIDER=local          # local | deepseek
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5

# Weather
WEATHER_PROVIDER=open_meteo       # open_meteo | qweather | gaode
QWEATHER_API_KEY=
GAODE_API_KEY=

# Memory
MAX_HISTORY_TURNS=20
REDIS_SESSION_TTL_SECONDS=86400

# Observability
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=http://langfuse:3000

# Security (optional)
API_KEY=

# Startup
SKIP_AUTO_INGEST=false
```

---

## 10. 技术栈

- Python 3.11+
- FastAPI + Uvicorn + Pydantic v2
- LangGraph + langchain-openai
- LlamaIndex + chromadb
- Redis (redis-py asyncio) + SQLAlchemy 2.0 async + Alembic + asyncpg
- Langfuse Python SDK
- prometheus-client（可选）
- pytest + pytest-asyncio + httpx
- Docker Compose

---

## 11. 非目标（第一版不做）

- 用户注册/登录/OAuth
- 多租户隔离
- Web 聊天 UI（仅 REST API）
- 法律文档实时爬取更新（仅脚本手动/CI 触发）
- 流式 SSE 响应（后续扩展）

---

## 12. 成功标准

1. `docker compose up` 一键启动全部服务
2. `/api/v1/chat` 能正确路由 legal / weather / general 三类问题
3. 法律问题返回答案 + citations + disclaimer
4. 多轮对话 memory 连贯（benchmark 验证）
5. `pytest tests/evaluation/ -m "not slow"` 在 CI 通过
6. Langfuse UI 可见完整 trace span 链
7. `profile/legal/` 含至少 4 份法律节选文档
