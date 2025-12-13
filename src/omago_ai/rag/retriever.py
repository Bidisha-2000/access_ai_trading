from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omago_ai.rag.chunking import chunk_documents
from omago_ai.rag.index import RagIndex, build_index, load_index, query_index, save_index
from omago_ai.rag.loader import load_corpus
from omago_ai.rag.types import RetrievedChunk


def _default_project_root() -> Path:
    # .../src/omago_ai/rag/retriever.py -> repo root is 3 parents up
    return Path(__file__).resolve().parents[3]


@dataclass
class RagConfig:
    top_k: int = 4


class RagRetriever:
    def __init__(
        self,
        *,
        project_root: Path | None = None,
        corpus_subdir: str = "data/corpus",
        cache_subdir: str = ".cache/rag",
        config: RagConfig | None = None,
    ) -> None:
        self.project_root = (project_root or _default_project_root()).resolve()
        self.corpus_dir = (self.project_root / corpus_subdir).resolve()
        self.cache_dir = (self.project_root / cache_subdir).resolve()
        self.config = config or RagConfig()

    @property
    def index_path(self) -> Path:
        # Bump this when the index format / chunk metadata changes.
        index_version = 2
        return self.cache_dir / f"index_v{index_version}.pkl"

    def get_index(self, *, rebuild: bool = False) -> RagIndex:
        if self.index_path.exists() and not rebuild:
            try:
                index_mtime = self.index_path.stat().st_mtime
                corpus_mtime = max((p.stat().st_mtime for p in self.corpus_dir.rglob(
                    "*") if p.is_file()), default=0)
                if corpus_mtime <= index_mtime:
                    return load_index(self.index_path)
            except Exception:
                # If anything looks odd, fall back to rebuilding.
                pass

        docs = load_corpus(self.corpus_dir, project_root=self.project_root)
        chunks = chunk_documents(docs)
        index = build_index(chunks)
        save_index(index, self.index_path)
        return index

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        index = self.get_index(rebuild=False)
        k = int(top_k) if top_k is not None else int(self.config.top_k)
        return query_index(index, query, top_k=k)
