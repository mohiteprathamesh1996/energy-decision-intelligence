# ── PuLP edition — CBC is bundled via pip, no apt installs needed ─────────────
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

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
    CMD python -c "import pulp; s=pulp.PULP_CBC_CMD(); print('ok')" || exit 1

ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app

CMD ["streamlit", "run", "app/dashboard.py", "--server.address=0.0.0.0"]