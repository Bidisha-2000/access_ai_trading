from __future__ import annotations

import re
from typing import Iterable

from omago_ai.rag.loader import LoadedDocument
from omago_ai.rag.types import DocumentChunk


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_documents(
    docs: list[LoadedDocument],
    *,
    max_chars: int = 900,
    overlap_chars: int = 120,
) -> list[DocumentChunk]:
    """Chunk documents into small pieces for retrieval.

    Uses paragraph-based splitting and then merges into max_chars windows.
    """

    chunks: list[DocumentChunk] = []

    for doc in docs:
        text = _normalize_whitespace(doc.text)
        paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

        buf: list[str] = []
        buf_len = 0

        def flush() -> None:
            nonlocal buf, buf_len
            if not buf:
                return
            joined = "\n\n".join(buf).strip()
            if joined:
                chunk_id = f"{doc.source_path}::chunk::{len(chunks)}"
                chunks.append(DocumentChunk(chunk_id=chunk_id,
                              source_path=doc.source_path, text=joined))
            buf = []
            buf_len = 0

        for para in paragraphs:
            if not para:
                continue

            if buf_len + len(para) + 2 <= max_chars:
                buf.append(para)
                buf_len += len(para) + 2
                continue

            # Flush current buffer and start new.
            flush()
            if len(para) <= max_chars:
                buf.append(para)
                buf_len = len(para)
            else:
                # Hard split long paragraphs.
                for piece in _sliding_window(para, max_chars=max_chars, overlap=overlap_chars):
                    chunk_id = f"{doc.source_path}::chunk::{len(chunks)}"
                    chunks.append(DocumentChunk(chunk_id=chunk_id,
                                  source_path=doc.source_path, text=piece))

        flush()

    return chunks


def _sliding_window(text: str, *, max_chars: int, overlap: int) -> Iterable[str]:
    start = 0
    step = max(1, max_chars - overlap)
    while start < len(text):
        end = min(len(text), start + max_chars)
        yield text[start:end].strip()
        if end == len(text):
            break
        start += step
