# Scoring API image.
# Prerequisite: train the model (models/lgbm.joblib) before building the image.
#   docker build -t credit-scoring-api .
#   docker run -p 8000:8000 credit-scoring-api
FROM python:3.12-slim

# LightGBM requires the OpenMP library
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /usr/local/bin/uv
WORKDIR /app

# Dependencies first: this layer stays cached as long as pyproject.toml and uv.lock do not change
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY models/lgbm.joblib ./models/lgbm.joblib
RUN uv sync --frozen --no-dev

ENV MODEL_PATH=/app/models/lgbm.joblib
EXPOSE 8000
# Non-root user
RUN useradd --create-home api && chown -R api /app
USER api

HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uv", "run", "--no-sync", "uvicorn", "credit_risk_scoring.api:app", "--host", "0.0.0.0", "--port", "8000"]
