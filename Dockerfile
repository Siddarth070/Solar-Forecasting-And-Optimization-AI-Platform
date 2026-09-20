# ── Base image ─────────────────────────────────────────────────
# Python 3.11 slim — small, fast, production-ready
FROM python:3.11-slim

# ── Metadata ───────────────────────────────────────────────────
LABEL maintainer="Siddharth Agrawal"
LABEL description="Solar Energy Forecasting & Grid Optimization Platform"
LABEL version="1.0.0"

# ── Set working directory ──────────────────────────────────────
WORKDIR /app

# ── Install system dependencies ────────────────────────────────
# These are needed by some Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Copy requirements first ────────────────────────────────────
# Docker caches this layer — only reinstalls if requirements change
COPY requirements.txt .

# ── Install Python dependencies ────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy project files ─────────────────────────────────────────
# No data/ here: it's gitignored (the training data is synthetic and
# regenerable, not shipped — see src/models/model_card.json) and copying
# a path that doesn't exist on a clean clone used to fail the build.
COPY src/ ./src/
COPY dashboard/ ./dashboard/
COPY configs/ ./configs/

# ── Non-root user ────────────────────────────────────────────────
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

# ── Expose ports (Streamlit dashboard + FastAPI) ────────────────
EXPOSE 8501 8000

# ── Health check ───────────────────────────────────────────────
# Overridden per-service in docker-compose.yml for the API container.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Default command: the standalone dashboard ───────────────────
# (what Streamlit Cloud / a single-container deploy runs). For both the
# API and the dashboard together, use `docker compose up` instead — see
# docker-compose.yml, which overrides this CMD for the api service.
CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
