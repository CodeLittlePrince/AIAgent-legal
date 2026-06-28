FROM node:20-slim AS web-builder

WORKDIR /web
COPY web/package.json web/package-lock.json* ./
RUN npm install
COPY web/ .
RUN npm run build

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/
COPY scripts/ scripts/
COPY alembic/ alembic/
COPY alembic.ini .

RUN pip install --no-cache-dir -e .

# Pre-download embedding + rerank models so runtime does not block on HuggingFace.
RUN python -c "from llama_index.embeddings.huggingface import HuggingFaceEmbedding; HuggingFaceEmbedding(model_name='BAAI/bge-small-zh-v1.5')"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base')"

COPY profile/ profile/
COPY --from=web-builder /web/dist /app/web/dist

EXPOSE 8000

CMD ["uvicorn", "legal_assistant.main:app", "--host", "0.0.0.0", "--port", "8000"]
