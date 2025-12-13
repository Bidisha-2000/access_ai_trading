from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    source_path: str
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: DocumentChunk
    score: float
