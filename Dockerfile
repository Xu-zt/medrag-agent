FROM python:3.11-slim

WORKDIR /app

# System dependencies for torch and BGE models
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from pyproject.toml
# Copy pyproject.toml + src first so this layer is cached unless deps change
COPY pyproject.toml .
COPY src/ ./src/
RUN pip install --no-cache-dir . && \
    pip install --no-cache-dir langchain-ollama

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "medrag.api.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
