from __future__ import annotations

import threading
from typing import Dict, List

from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from sentence_transformers import SentenceTransformer

from .models import EmbedRequest, EmbedResponse


def create_app(
    model: SentenceTransformer,
    model_name: str,
    max_inference_concurrency: int = 1,
) -> FastAPI:
    app = FastAPI(
        title="Embeddings Service",
        version="1.0.0",
        default_response_class=ORJSONResponse,
    )
    effective_concurrency = max(1, max_inference_concurrency)
    inference_slots = threading.Semaphore(effective_concurrency)

    @app.get("/health")
    def health() -> Dict[str, object]:
        return {
            "ok": True,
            "model_loaded": True,
            "model": model_name,
            "max_inference_concurrency": effective_concurrency,
        }

    @app.post("/embed", response_model=EmbedResponse)
    def embed(req: EmbedRequest) -> EmbedResponse:
        texts = req.texts if isinstance(req.texts, list) else [req.texts]
        batch_size = min(req.batch_size, max(1, len(texts)))

        with inference_slots:
            previous_max_seq_length = None
            if req.max_length is not None:
                previous_max_seq_length = model.max_seq_length
                model.max_seq_length = req.max_length

            try:
                vecs = model.encode(
                    texts,
                    batch_size=batch_size,
                    normalize_embeddings=req.normalize,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            finally:
                if previous_max_seq_length is not None:
                    model.max_seq_length = previous_max_seq_length

        embeddings: List[List[float]] = vecs.tolist()
        dim = len(embeddings[0]) if embeddings else 0
        return EmbedResponse(model=model_name, dim=dim, embeddings=embeddings)

    return app
