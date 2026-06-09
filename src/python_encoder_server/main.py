from __future__ import annotations

import os
import time
from logging import getLogger

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

from .api import create_app as create_api_app
from .log_config import configure_logging

logger = getLogger(__name__)

MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR")
DEFAULT_VCPU_COUNT = 2


def _int_env(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def configure_torch_threads() -> None:
    vcpu_count = _int_env("VCPU_COUNT", DEFAULT_VCPU_COUNT, minimum=1)
    torch_threads = _int_env("TORCH_NUM_THREADS", vcpu_count, minimum=1)
    torch_interop_threads = _int_env("TORCH_NUM_INTEROP_THREADS", 1, minimum=1)
    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(torch_interop_threads)


def load_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    started_at = time.perf_counter()
    model = SentenceTransformer(model_name, cache_folder=MODEL_CACHE_DIR, device="cpu")
    logger.info(
        "model loaded",
        extra={
            "event": "model_loaded",
            "model": model_name,
            "model_cache_dir": MODEL_CACHE_DIR,
            "load_duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
        },
    )
    return model


def create_app() -> FastAPI:
    configure_logging()
    configure_torch_threads()
    model = load_model(MODEL_NAME)
    max_concurrency = _int_env("MAX_INFERENCE_CONCURRENCY", 1, minimum=1)
    return create_api_app(
        model=model,
        model_name=MODEL_NAME,
        max_inference_concurrency=max_concurrency,
    )


app = create_app()
