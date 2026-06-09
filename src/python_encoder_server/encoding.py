from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

from sentence_transformers import SentenceTransformer


class ModelEncoder:
    def __init__(
        self,
        model: SentenceTransformer,
        max_inference_concurrency: int = 1,
    ) -> None:
        self._model = model
        self.max_inference_concurrency = max(1, max_inference_concurrency)
        self._inference_slots = threading.Semaphore(self.max_inference_concurrency)
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
        with self._inference_slots:
            if max_length is None:
                with self._read_model_state():
                    return self._encode(
                        texts,
                        batch_size=batch_size,
                        normalize_embeddings=normalize_embeddings,
                    )

            with self._write_model_state():
                previous_max_seq_length = self._model.max_seq_length
                self._model.max_seq_length = max_length
                try:
                    return self._encode(
                        texts,
                        batch_size=batch_size,
                        normalize_embeddings=normalize_embeddings,
                    )
                finally:
                    self._model.max_seq_length = previous_max_seq_length

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
