FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    JEONSE_LLM=api \
    LLM_PROVIDER=openai

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

RUN useradd --create-home --uid 10001 appuser

COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser models ./models
COPY --chown=appuser:appuser data/generated ./data/generated
COPY --chown=appuser:appuser data/processed/owner_asset_ratio/buildings.csv ./data/processed/owner_asset_ratio/buildings.csv
COPY --chown=appuser:appuser data/downloaded/safety ./data/downloaded/safety
COPY --chown=appuser:appuser data/downloaded/facilities ./data/downloaded/facilities
COPY --chown=appuser:appuser data/downloaded/finance_policies ./data/downloaded/finance_policies
COPY --chown=appuser:appuser data/downloaded/finance_products ./data/downloaded/finance_products

# 기존 10만 건 부동산 DB는 유지하고 배포 시 금융 테이블만 최신 소스로 재구성한다.
RUN python -c "import sqlite3; from src import config; from src.db.build_db import build_finance_db; c=sqlite3.connect(config.DB_PATH); build_finance_db(c); c.close()"

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

CMD ["uvicorn", "src.server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips=*"]
