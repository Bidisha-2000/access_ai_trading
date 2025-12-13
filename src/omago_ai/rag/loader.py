from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LoadedDocument:
    source_path: str
    text: str


def load_corpus(corpus_dir: Path, *, project_root: Path | None = None) -> list[LoadedDocument]:
    """Load plaintext corpus files from disk.

    Supports: .md, .txt
    """

    corpus_dir = corpus_dir.resolve()
    if not corpus_dir.exists():
        return []

    root = project_root.resolve() if project_root else None

    docs: list[LoadedDocument] = []
    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".md", ".txt"}:
            continue

        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue

        if root is not None:
            try:
                source_path = str(path.resolve().relative_to(root))
            except Exception:
                source_path = str(path.relative_to(corpus_dir))
        else:
            source_path = str(path.relative_to(corpus_dir))

        docs.append(LoadedDocument(source_path=source_path, text=text))

    return docs
