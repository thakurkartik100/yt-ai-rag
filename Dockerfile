# syntax=docker/dockerfile:1
# Build a lean production image for the Tube AI RAG API.
#
# Build:  docker build -t yt-ai-rag .
# Run:    docker run -p 8000:8000 --env-file .env yt-ai-rag

FROM python:3.12-slim

WORKDIR /app

# System libs needed by sentence-transformers / chromadb
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer-cached separately from code)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/ ./app/

EXPOSE 8000

# Health check -- Docker Desktop shows green when the API is up
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
