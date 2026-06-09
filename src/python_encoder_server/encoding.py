from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from logging import getLogger
from typing import Any, Iterator, Sequence

from sentence_transformers import SentenceTransformer

logger = getLogger(__name__)


class ModelEncoder:
    def __init__(
        self,
        model: SentenceTransformer,
        max_inference_concurrency: int = 1,
        transport: str = "unknown",
    ) -> None:
        self._model = model
        self._transport = transport
        self.max_inference_concurrency = max(1, max_inference_concurrency)
        self._inference_slots = threading.Semaphore(self.max_inference_concurrency)
        self._concurrency_lock = threading.Lock()
        self._active_encodes = 0
        self._queued_encodes = 0
        self._state_condition = threading.Condition()
        self._active_state_readers = 0
        self._pending_state_writers = 0
        self._state_writer_active = False

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        max_length: int | None = None,
    ) -> Any:
        queue_started_at = time.perf_counter()
        active_at_enqueue, queued_at_enqueue = self._enqueue()
        self._inference_slots.acquire()
        wait_seconds = time.perf_counter() - queue_started_at
        active_at_start, queued_at_start = self._start_encode()
        encode_started_at = time.perf_counter()
        try:
            if max_length is None:
                with self._read_model_state():
                    result = self._encode(
                        texts,
                        batch_size=batch_size,
                        normalize_embeddings=normalize_embeddings,
                    )
            else:
                with self._write_model_state():
                    previous_max_seq_length = self._model.max_seq_length
                    self._model.max_seq_length = max_length
                    try:
                        result = self._encode(
                            texts,
                            batch_size=batch_size,
                            normalize_embeddings=normalize_embeddings,
                        )
                    finally:
                        self._model.max_seq_length = previous_max_seq_length

            self._log_encode_complete(
                result,
                texts=texts,
                batch_size=batch_size,
                normalize_embeddings=normalize_embeddings,
                max_length=max_length,
                wait_seconds=wait_seconds,
                encode_seconds=time.perf_counter() - encode_started_at,
                active_at_enqueue=active_at_enqueue,
                queued_at_enqueue=queued_at_enqueue,
                active_at_start=active_at_start,
                queued_at_start=queued_at_start,
            )
            return result
        finally:
            self._finish_encode()
            self._inference_slots.release()

    def _encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
    ) -> Any:
        return self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def _enqueue(self) -> tuple[int, int]:
        with self._concurrency_lock:
            active_at_enqueue = self._active_encodes
            queued_at_enqueue = self._queued_encodes
            self._queued_encodes += 1
            return active_at_enqueue, queued_at_enqueue

    def _start_encode(self) -> tuple[int, int]:
        with self._concurrency_lock:
            self._queued_encodes -= 1
            self._active_encodes += 1
            return self._active_encodes, self._queued_encodes

    def _finish_encode(self) -> None:
        with self._concurrency_lock:
            self._active_encodes -= 1

    def _log_encode_complete(
        self,
        result: Any,
        *,
        texts: Sequence[str],
        batch_size: int,
        normalize_embeddings: bool,
        max_length: int | None,
        wait_seconds: float,
        encode_seconds: float,
        active_at_enqueue: int,
        queued_at_enqueue: int,
        active_at_start: int,
        queued_at_start: int,
    ) -> None:
        shape = getattr(result, "shape", None)
        embedding_count = (
            int(shape[0]) if shape is not None and len(shape) > 0 else len(result)
        )
        embedding_dim = int(shape[1]) if shape is not None and len(shape) > 1 else None

        logger.info(
            "embedding encode completed",
            extra={
                "event": "embedding_encode_completed",
                "transport": self._transport,
                "text_count": len(texts),
                "embedding_count": embedding_count,
                "embedding_dim": embedding_dim,
                "batch_size": batch_size,
                "normalize_embeddings": normalize_embeddings,
                "max_length_set": max_length is not None,
                "max_inference_concurrency": self.max_inference_concurrency,
                "active_at_enqueue": active_at_enqueue,
                "queued_at_enqueue": queued_at_enqueue,
                "active_at_start": active_at_start,
                "queued_at_start": queued_at_start,
                "queue_wait_ms": round(wait_seconds * 1000, 3),
                "encode_duration_ms": round(encode_seconds * 1000, 3),
            },
        )

    @contextmanager
    def _read_model_state(self) -> Iterator[None]:
        with self._state_condition:
            while self._state_writer_active or self._pending_state_writers > 0:
                self._state_condition.wait()
            self._active_state_readers += 1
        try:
            yield
        finally:
            with self._state_condition:
                self._active_state_readers -= 1
                if self._active_state_readers == 0:
                    self._state_condition.notify_all()

    @contextmanager
    def _write_model_state(self) -> Iterator[None]:
        with self._state_condition:
            self._pending_state_writers += 1
            while self._state_writer_active or self._active_state_readers > 0:
                self._state_condition.wait()
            self._pending_state_writers -= 1
            self._state_writer_active = True
        try:
            yield
        finally:
            with self._state_condition:
                self._state_writer_active = False
                self._state_condition.notify_all()
