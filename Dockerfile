# Punar -- autonomous recovery agent for failed Razorpay payments.
# Multi-stage build: wheels are compiled once, the runtime image stays slim
# and runs as an unprivileged user.

FROM python:3.11-slim AS builder

WORKDIR /build
RUN python -m pip install --no-cache-dir --upgrade pip build

COPY pyproject.toml requirements.txt README.md ./
COPY punar ./punar
RUN python -m build --wheel --outdir /dist


FROM python:3.11-slim AS runtime

# Fail fast, no .pyc, unbuffered logs so the container streams them straight out.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# The audit trail is a compliance record: it lives on a volume, not in the layer.
ENV PUNAR_DB_PATH=/data/punar_audit.db \
    PUNAR_BANDIT_DB=/data/punar_bandit.db

RUN groupadd --system punar && useradd --system --gid punar --home /app punar

WORKDIR /app
COPY --from=builder /dist/*.whl /tmp/
RUN python -m pip install --no-cache-dir /tmp/*.whl && rm -f /tmp/*.whl

RUN mkdir -p /data && chown -R punar:punar /data /app
USER punar
VOLUME ["/data"]

EXPOSE 8000

# /health checks real dependencies (policy loadable, audit DB writable),
# so an unhealthy container is one that genuinely cannot serve.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status==200 else 1)"

CMD ["uvicorn", "punar.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
