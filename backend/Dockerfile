# Build context: repository root (the app imports `backend.app.*` and `scripts.*`).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEEPEVAL_TELEMETRY_OPT_OUT=YES

# libgomp1: OpenMP runtime needed by XGBoost on Linux
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Optional extra root CAs for TLS-intercepting proxies (see docker/certs/README.md)
COPY docker/certs/ /usr/local/share/ca-certificates/extra/
RUN update-ca-certificates
ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend backend
COPY scripts scripts

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -fsS http://localhost:8000/api/v1/health > /dev/null || exit 1

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
