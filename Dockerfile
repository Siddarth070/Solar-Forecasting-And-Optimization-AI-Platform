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
COPY src/ ./src/
COPY dashboard/ ./dashboard/
COPY configs/ ./configs/
COPY data/ ./data/

# ── Expose port ────────────────────────────────────────────────
EXPOSE 8501

# ── Health check ───────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Run the dashboard ──────────────────────────────────────────
CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]