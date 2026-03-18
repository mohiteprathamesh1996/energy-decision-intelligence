# ─── Stage 1: solver build ────────────────────────────────────────────────────
FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
        coinor-cbc \
        coinor-libcbc-dev \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Stage 2: application ────────────────────────────────────────────────────
FROM base AS app

COPY . .

# Streamlit config
RUN mkdir -p /root/.streamlit && printf '\
[server]\n\
headless = true\n\
port = 8501\n\
enableCORS = false\n\
[theme]\n\
base = "dark"\n\
backgroundColor = "#0d1117"\n\
secondaryBackgroundColor = "#161b22"\n\
textColor = "#c9d1d9"\n\
primaryColor = "#58a6ff"\n' > /root/.streamlit/config.toml

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Default: launch dashboard. Override with: docker run ... python run.py --mode full
CMD ["streamlit", "run", "app/dashboard.py", "--server.address=0.0.0.0"]
