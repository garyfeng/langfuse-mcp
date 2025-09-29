FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    LANGFUSE_PUBLIC_KEY="" \
    LANGFUSE_SECRET_KEY="" \
    LANGFUSE_HOST="https://cloud.langfuse.com" \
    LANGFUSE_LOG_LEVEL="INFO" \
    LANGFUSE_LOG_TO_CONSOLE="false"

RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY langfuse_mcp /app/langfuse_mcp

RUN pip install --no-cache-dir .

USER appuser

ENTRYPOINT ["python", "-m", "langfuse_mcp"]
