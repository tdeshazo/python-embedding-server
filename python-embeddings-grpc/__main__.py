from __future__ import annotations

import os
import threading
from concurrent import futures
from typing import List

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import grpc
import torch
from sentence_transformers import SentenceTransformer

import embeddings_pb2
import embeddings_pb2_grpc

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


class EmbeddingService(embeddings_pb2_grpc.EmbeddingServiceServicer):
    def __init__(
        self,
        model: SentenceTransformer,
        model_name: str,
        max_inference_concurrency: int = 1,
    ) -> None:
        self._model = model
        self._model_name = model_name
        self._effective_concurrency = max(1, max_inference_concurrency)
        self._inference_slots = threading.Semaphore(self._effective_concurrency)

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
            max_inference_concurrency=self._effective_concurrency,
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

        with self._inference_slots:
            previous_max_seq_length = None
            if request.HasField("max_length"):
                previous_max_seq_length = self._model.max_seq_length
                self._model.max_seq_length = request.max_length

            try:
                vecs = self._model.encode(
                    texts,
                    batch_size=effective_batch_size,
                    normalize_embeddings=normalize,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            finally:
                if previous_max_seq_length is not None:
                    self._model.max_seq_length = previous_max_seq_length

        vectors: List[List[float]] = vecs.tolist()
        dim = len(vectors[0]) if vectors else 0
        embeddings = [embeddings_pb2.Embedding(values=row) for row in vectors]
        return embeddings_pb2.EmbedResponse(
            model=self._model_name,
            dim=dim,
            embeddings=embeddings,
        )


def serve() -> None:
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
