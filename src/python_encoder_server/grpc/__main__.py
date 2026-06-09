from __future__ import annotations

import os
import time
from concurrent import futures
from logging import getLogger
from typing import List

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import grpc
import torch
from sentence_transformers import SentenceTransformer

from python_encoder_server.encoding import ModelEncoder
from python_encoder_server.log_config import configure_logging

from . import embeddings_pb2, embeddings_pb2_grpc

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


class EmbeddingService(embeddings_pb2_grpc.EmbeddingServiceServicer):
    def __init__(
        self,
        model: SentenceTransformer,
        model_name: str,
        max_inference_concurrency: int = 1,
    ) -> None:
        self._model_name = model_name
        self._encoder = ModelEncoder(
            model=model,
            max_inference_concurrency=max_inference_concurrency,
            transport="grpc",
        )

    def Health(
        self,
        request: embeddings_pb2.HealthRequest,
        context: grpc.ServicerContext,
    ) -> embeddings_pb2.HealthResponse:
        del request
        del context
        return embeddings_pb2.HealthResponse(
            ok=True,
            model_loaded=True,
            model=self._model_name,
            max_inference_concurrency=self._encoder.max_inference_concurrency,
        )

    def Embed(
        self,
        request: embeddings_pb2.EmbedRequest,
        context: grpc.ServicerContext,
    ) -> embeddings_pb2.EmbedResponse:
        texts = list(request.texts)
        normalize = request.normalize if request.HasField("normalize") else True
        batch_size = request.batch_size if request.HasField("batch_size") else 16

        if batch_size < 1 or batch_size > 256:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "batch_size must be between 1 and 256",
            )

        effective_batch_size = min(batch_size, max(1, len(texts)))

        vecs = self._encoder.encode(
            texts,
            batch_size=effective_batch_size,
            normalize_embeddings=normalize,
            max_length=request.max_length if request.HasField("max_length") else None,
        )

        vectors: List[List[float]] = vecs.tolist()
        dim = len(vectors[0]) if vectors else 0
        embeddings = [embeddings_pb2.Embedding(values=row) for row in vectors]
        return embeddings_pb2.EmbedResponse(
            model=self._model_name,
            dim=dim,
            embeddings=embeddings,
        )


def serve() -> None:
    configure_logging()
    configure_torch_threads()
    model = load_model(MODEL_NAME)
    max_inference_concurrency = _int_env("MAX_INFERENCE_CONCURRENCY", 1, minimum=1)
    max_workers = _int_env("GRPC_MAX_WORKERS", max_inference_concurrency, minimum=1)
    bind_addr = os.getenv("GRPC_BIND_ADDR", "0.0.0.0:50051")
    shutdown_grace_seconds = _int_env("GRPC_SHUTDOWN_GRACE_SECONDS", 10, minimum=0)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    embeddings_pb2_grpc.add_EmbeddingServiceServicer_to_server(
        EmbeddingService(
            model=model,
            model_name=MODEL_NAME,
            max_inference_concurrency=max_inference_concurrency,
        ),
        server,
    )
    server.add_insecure_port(bind_addr)
    server.start()
    print(f"gRPC embedding service listening on {bind_addr}")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print(
            "Ctrl+C received, shutting down gRPC server "
            f"(grace={shutdown_grace_seconds}s)..."
        )
        server.stop(grace=shutdown_grace_seconds)
        server.wait_for_termination(timeout=shutdown_grace_seconds + 1)
        print("gRPC server stopped.")


if __name__ == "__main__":
    serve()
