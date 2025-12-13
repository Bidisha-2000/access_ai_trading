from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from omago_ai.rag.types import DocumentChunk, RetrievedChunk


@dataclass
class RagIndex:
    vectorizer: TfidfVectorizer
    matrix: "np.ndarray | object"  # sparse matrix
    chunks: list[DocumentChunk]


def build_index(chunks: list[DocumentChunk]) -> RagIndex:
    # Word-level TF-IDF works well for a small corpus and runs offline.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=40_000,
    )
    matrix = vectorizer.fit_transform([c.text for c in chunks])
    return RagIndex(vectorizer=vectorizer, matrix=matrix, chunks=chunks)


def save_index(index: RagIndex, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(index, f)


def load_index(path: Path) -> RagIndex:
    with path.open("rb") as f:
        return pickle.load(f)


def query_index(index: RagIndex, query: str, *, top_k: int = 5) -> list[RetrievedChunk]:
    q = (query or "").strip()
    if not q:
        return []

    q_vec = index.vectorizer.transform([q])
    # Cosine similarity because TF-IDF vectors are L2-normalized by default.
    scores = (index.matrix @ q_vec.T).toarray().reshape(-1)

    if scores.size == 0:
        return []

    top_k = max(1, int(top_k))
    idxs = np.argsort(scores)[::-1][:top_k]

    results: list[RetrievedChunk] = []
    for i in idxs:
        score = float(scores[i])
        if score <= 0:
            continue
        results.append(RetrievedChunk(chunk=index.chunks[int(i)], score=score))

    return results
