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
