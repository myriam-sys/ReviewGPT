FROM python:3.11-slim

WORKDIR /app

# System dependencies needed for sentence-transformers and asyncpg
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Put the HuggingFace cache inside /app so it stays under the directory
# we chown to appuser later. Without this, the model downloads to
# /root/.cache/ which the non-root user cannot read at runtime.
ENV HF_HOME=/app/.cache/huggingface

# Pre-download embedding model during build so Render cold starts don't
# trigger a 420 MB download at request time (which causes timeouts).
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'); \
print('Model pre-downloaded successfully')"

# Copy backend source code
COPY backend/ ./backend/

# Ensure Python can resolve 'backend.*' imports from /app.
# WORKDIR is /app, so 'backend' is a direct child directory.
# Setting PYTHONPATH makes this explicit and independent of how
# uvicorn is invoked (avoids ModuleNotFoundError on Render).
ENV PYTHONPATH=/app

# Create non-root user for security
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check — Render also calls /health via healthCheckPath in render.yaml
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# Single worker — the sentence-transformers singleton is not fork-safe
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
