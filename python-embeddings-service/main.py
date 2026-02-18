from __future__ import annotations

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

from api import create_app as create_api_app

MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
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
    return SentenceTransformer(model_name, device="cpu")


def create_app() -> FastAPI:
    configure_torch_threads()
    model = load_model(MODEL_NAME)
    max_concurrency = _int_env("MAX_INFERENCE_CONCURRENCY", 1, minimum=1)
    return create_api_app(
        model=model,
        model_name=MODEL_NAME,
        max_inference_concurrency=max_concurrency,
    )


app = create_app()
