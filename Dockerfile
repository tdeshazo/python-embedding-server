FROM python:3.11-slim

ARG MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_NAME=${MODEL_NAME} \
    MODEL_CACHE_DIR=/opt/model-cache/sentence-transformers \
    HF_HOME=/opt/model-cache/huggingface \
    XDG_CACHE_HOME=/opt/model-cache \
    VCPU_COUNT=2 \
    TORCH_NUM_THREADS=2 \
    TORCH_NUM_INTEROP_THREADS=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    TOKENIZERS_PARALLELISM=false \
    MAX_INFERENCE_CONCURRENCY=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 tini \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY src /app/src
RUN pip install --upgrade pip \
    && pip install ".[json]"

RUN mkdir -p "${MODEL_CACHE_DIR}" "${HF_HOME}" \
    && python -c "import os; from sentence_transformers import SentenceTransformer; SentenceTransformer(os.environ['MODEL_NAME'], cache_folder=os.environ['MODEL_CACHE_DIR'], device='cpu')" \
    && chmod -R a+rX /opt/model-cache

RUN useradd -m -u 10001 appuser \
    && chown -R appuser:appuser /opt/model-cache
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "python_encoder_server.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "8", "--timeout-keep-alive", "5", "--loop", "uvloop", "--http", "httptools"]
