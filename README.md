# 智能法律助手（Legal Assistant）

面向生产场景的对话式 AI REST 服务：**中文法律咨询**（RAG）、**天气查询**与**通用闲聊**，基于 LangGraph + DeepSeek Tool Calling 构建。法律知识库使用 LlamaIndex + Chroma 向量检索，配合 Cross-Encoder 精排；会话记忆为 Redis + PostgreSQL；可观测性接入 Langfuse 与 Prometheus。

## 功能概览

| 能力 | 说明 |
|------|------|
| 法律咨询 | Agent 调用 `search_legal_knowledge`，基于检索片段回答并附引用与免责声明 |
| 天气查询 | 规则识别常见问法 → 快速路径直调天气 API；其余走 Agent 工具 `get_weather_forecast` |
| 多轮对话 | Redis 热缓存 + PostgreSQL 持久化，支持 `session_id` 续聊 |
| 流式输出 | `POST /api/v1/chat/stream`（SSE），React 前端实时展示 |
| 可观测性 | Langfuse v3 链路追踪；`/metrics` Prometheus 指标 |

## 架构

混合路由：**天气走规则快速路径，法律与其余意图走 Tool Calling Agent**。

```text
START → route（weather_rules）→ weather | agent（Tool Calling 循环）→ save_memory → END
```

| 路径 | 触发条件 | 行为 |
|------|----------|------|
| **weather** | `planner/weather_rules.detect_weather_route()` 命中 | 直调天气 adapter → 单次 LLM 组织回复 |
| **agent** | 其余所有请求 | `bind_tools` 循环，可用工具见下表 |

Agent 可用工具（`tools/builder.py`）：

| 工具名 | 模块 | 用途 |
|--------|------|------|
| `search_legal_knowledge` | `tools/legal/search.py` | 封装 `knowledge/retriever`，RAG 检索法条 |
| `get_weather_forecast` | `tools/weather/forecast.py` | 封装天气 adapter（Open-Meteo / 和风 / 高德） |

### RAG 检索（粗筛 + Rerank）

`knowledge/retriever.py` 采用两阶段检索，避免口语化问法在向量相似度阈值阶段被过早过滤：

```text
用户 query
  → 向量粗筛：Chroma 召回 RAG_COARSE_TOP_K（默认 20）条，不做硬阈值过滤
  → Cross-Encoder 精排：BAAI/bge-reranker-base
  → 取 RAG_FINAL_TOP_K（默认 5）条返回 Agent
```

嵌入模型默认 `BAAI/bge-small-zh-v1.5`。Docker 构建阶段会预下载 embedding 与 rerank 权重；本地开发可手动预热（见下文）。

## 环境要求

- **Docker** + **Docker Compose** v2+
- **Python 3.11+**（本地开发与测试）
- **`DEEPSEEK_API_KEY`** — LLM 对话、评测与 benchmark 必需（[DeepSeek 开放平台](https://platform.deepseek.com/)）

复制环境变量模板并填写密钥：

```bash
cp .env.example .env
# 至少设置 DEEPSEEK_API_KEY
# 启用 Langfuse 追踪时设置 LANGFUSE_PUBLIC_KEY、LANGFUSE_SECRET_KEY
```

## 快速开始（Docker Compose）

启动完整栈（API、Postgres、Redis、Chroma、Langfuse）：

```bash
docker compose up -d --build
```

首次启动 API 健康检查 `start_period` 较长（需加载 RAG 模型），可查看状态：

```bash
docker compose ps
curl -s http://localhost:8000/health | jq .
```

Docker 中 API 默认 `SKIP_AUTO_INGEST=true`，知识库为空时需手动入库：

```bash
curl -s -X POST http://localhost:8000/api/v1/knowledge/reindex | jq .
```

### 服务与端口

| 服务 | 端口 | 说明 |
|------|------|------|
| api | 8000 | FastAPI（含 React 静态页 `/`） |
| postgres | 5432 | 会话与消息持久化 |
| redis | 6379 | 会话热缓存 |
| chroma | 8001 | 法律知识向量库 |
| langfuse-web | 3000 | Langfuse v3 控制台 |
| langfuse-worker | — | Langfuse 后台写入 |
| langfuse-clickhouse | 8123 / 9000 | Langfuse 分析存储 |
| langfuse-minio | — | Langfuse 对象存储 |

停止：

```bash
docker compose down
```

## 本地开发

### 1. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
# 或使用 uv：uv sync
```

### 2. 预下载 RAG 模型（推荐）

embedding 与 rerank 模型体积较大，建议在首次启动前下载，避免第一次法律咨询阻塞：

```bash
python scripts/download_rag_models.py
```

也可在 `.env` 中设置 `RAG_WARMUP_ON_STARTUP=true`，让 API 启动时自动预加载（启动变慢，首次检索更快）。

### 3. 启动基础设施

Postgres、Redis、Chroma 为记忆与 RAG 所必需：

```bash
docker compose up -d postgres redis chroma
```

本地跑 API 时，`.env` 中主机名用 `localhost`（不要用 Docker 服务名 `postgres` / `chroma`）：

```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/legal_assistant
REDIS_URL=redis://localhost:6379/0
CHROMA_HOST=localhost
CHROMA_PORT=8001
```

### 4. 启动 API

```bash
uvicorn legal_assistant.main:app --reload --port 8000
```

### 5. 启动前端（开发模式）

```bash
cd web && npm install && npm run dev
```

浏览器打开 **http://localhost:5173**（Vite 代理到 API）。

生产构建由 FastAPI 托管 `web/dist`，访问 **http://localhost:8000/** 即可；`docker compose up --build` 会在镜像内完成前端构建。

## API 示例

### 法律咨询

```bash
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"劳动合同试用期最长多久？"}' | jq .
```

期望：`tools_used` 含 `search_legal_knowledge`，`citations` 非空，带 `disclaimer`。

### 天气

```bash
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"上海今天天气怎么样？"}' | jq .
```

期望：`tools_used` 含 `get_weather_forecast`（或走 weather 快速路径时同样标记该工具）。

### 闲聊

```bash
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"你好，介绍一下你自己"}' | jq .
```

期望：`tools_used` 为空数组 `[]`。

### 流式对话（SSE）

```bash
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"message":"你好"}'
```

事件顺序：`session` → `status` → `tools` → `delta`（正文分块）→ `citations` / `disclaimer` → `done`。

同步 JSON 接口 `POST /api/v1/chat` 仍可用于非流式客户端。

### 健康检查与指标

```bash
curl -s http://localhost:8000/health | jq .
curl -s http://localhost:8000/metrics
```

## 主要配置项

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `DEEPSEEK_API_KEY` | — | DeepSeek API 密钥 |
| `DEEPSEEK_MODEL` | `deepseek-chat` | 对话模型 |
| `AGENT_MAX_TOOL_ROUNDS` | `4` | Agent 每轮最多工具调用迭代次数 |
| `EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 向量嵌入模型 |
| `RAG_COARSE_TOP_K` | `20` | 向量粗筛召回数 |
| `RAG_FINAL_TOP_K` | `5` | rerank 后返回条数 |
| `RAG_RERANK_MODEL` | `BAAI/bge-reranker-base` | Cross-Encoder 精排模型 |
| `RAG_WARMUP_ON_STARTUP` | `true` | 启动时预加载 RAG 模型 |
| `WEATHER_PROVIDER` | `open_meteo` | 天气数据源（`qweather` / `gaode` 需对应 API Key） |
| `SKIP_AUTO_INGEST` | `false` | 为 `true` 时跳过启动自动入库 |
| `LANGFUSE_ENABLED` | `false` | 是否上报 Langfuse 追踪 |

完整列表见 [`.env.example`](.env.example)。

## 测试

```bash
# 单元测试
pytest tests/unit -v --tb=short

# 集成测试
pytest tests/integration -v --tb=short

# 评测（RAG 召回、LLM Judge、Agent Benchmark）
pytest tests/evaluation -v --tb=short

# CI 快速套件（排除 slow / benchmark）
pytest tests/unit tests/integration tests/evaluation -m "not slow and not benchmark" -v --tb=short
```

需调用真实 LLM 的慢测与 benchmark（须配置 `DEEPSEEK_API_KEY`）：

```bash
pytest tests/evaluation -m slow -v --tb=short
pytest tests/evaluation/test_agent_benchmark.py -m benchmark -v --tb=short
```

## Langfuse

启动 Langfuse v3 栈后访问 **http://localhost:3000**：

```bash
docker compose up -d langfuse-db langfuse-redis langfuse-clickhouse langfuse-minio langfuse-web langfuse-worker
```

在 UI 创建项目，将 Public/Secret Key 写入 `.env`（`LANGFUSE_ENABLED=true`，`LANGFUSE_HOST=http://localhost:3000`），重启 API 即可在 Langfuse 中查看对话 trace。

> **从 Langfuse v2 升级：** v3 数据库 schema 不同，需重置 Langfuse Postgres volume 后重新注册。

## 源码模块

| 模块 | 路径 | 职责 |
|------|------|------|
| API | `src/legal_assistant/api/` | REST 路由、SSE、`chat_service` |
| Runtime | `src/legal_assistant/runtime/` | LangGraph 图、节点、Agent Tool Calling 循环 |
| Planner | `src/legal_assistant/planner/` | 天气规则路由（`weather_rules.py`） |
| Knowledge | `src/legal_assistant/knowledge/` | 文档入库、Chroma、粗筛检索、rerank、模型预热 |
| Tools | `src/legal_assistant/tools/` | Agent 工具与天气 adapter |
| Memory | `src/legal_assistant/memory/` | Redis + PostgreSQL 会话记忆 |
| Observability | `src/legal_assistant/observability/` | Langfuse、Prometheus |
| Evaluation | `src/legal_assistant/evaluation/` | RAG 指标、LLM Judge、Agent Benchmark |

更细的 `tools/` 与请求链路说明见 [`src/legal_assistant/README.md`](src/legal_assistant/README.md)。

## 实用脚本

| 脚本 | 用途 |
|------|------|
| `scripts/download_rag_models.py` | 预下载 embedding + rerank 模型 |
| `scripts/ingest_knowledge.py` | 命令行入库法律文档 |
| `scripts/download_legal_docs.py` | 下载/更新 `profile/legal/` 静态法条 |
| `scripts/run_agent_benchmark.py` | 运行 Agent 端到端 benchmark |
| `scripts/export_low_score_traces.py` | 从 Langfuse 导出低分 trace 便于分析 |

## 非目标（v1）

依据 [设计文档 §11](docs/superpowers/specs/2025-06-24-intelligent-ai-assistant-design.md#11-非目标第一版不做)，首版仍不包含：

| 不做 | v1 替代方案 |
|------|-------------|
| 用户注册 / 登录 / OAuth | 无鉴权端点；可选 `API_KEY` 预留给后续中间件 |
| 多租户隔离 | 仅按 `session_id` 区分会话 |
| 法条实时爬取 | 手动或 CI 维护 `profile/legal/` + `POST /api/v1/knowledge/reindex` |

**已包含：** React Web 聊天 UI（`web/`，`/ ` 托管）、SSE 流式接口、Tool Calling Agent 混合架构。

合规测试：`tests/unit/test_non_goals.py`；Web/SSE：`tests/unit/test_web_features.py`、`tests/integration/test_api_chat.py`。

## 项目结构

```text
src/legal_assistant/   # Python 应用包
web/                   # React 聊天前端（Vite + TypeScript）
profile/legal/         # 静态法律 Markdown 文档
tests/unit/            # 单元测试
tests/integration/     # API 集成测试
tests/evaluation/      # RAG / Judge / Benchmark 评测
scripts/               # 入库、模型下载、trace 导出等
alembic/               # 数据库迁移
docs/                  # 设计与实现文档
```
