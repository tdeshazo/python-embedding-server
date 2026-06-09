from __future__ import annotations

from typing import List, Optional, Union

from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    texts: Union[str, List[str]] = Field(
        ..., description="A string or a list of strings."
    )
    normalize: bool = Field(
        True, description="L2-normalize embeddings (recommended for cosine)."
    )
    batch_size: int = Field(16, ge=1, le=256)
    max_length: Optional[int] = Field(
        None, description="Override max sequence length if set."
    )


class EmbedResponse(BaseModel):
    model: str
    dim: int
    embeddings: List[List[float]]
