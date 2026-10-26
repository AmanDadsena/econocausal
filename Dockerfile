# EconoCausal API
#
#   docker build -t econocausal-api .
#   docker run -p 8000:8000 -v ./data:/app/data:ro econocausal-api
#
# The SQLite database is not baked into the image: mount the folder that contains
# econocausal.db at /app/data (it is built by the pipeline, see README "How to run").

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ECONOCAUSAL_DB=/app/data/econocausal.db

WORKDIR /app

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY pyproject.toml .
COPY src ./src
RUN pip install --no-deps .

COPY api ./api

# Run as an unprivileged user.
RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:' + __import__('os').environ.get('PORT', '8000') + '/health', timeout=4)"

# PORT is set by most hosting platforms; 8000 locally.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
