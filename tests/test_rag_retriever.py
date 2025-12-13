from __future__ import annotations

from pathlib import Path

from omago_ai.rag.retriever import RagRetriever


def test_rag_retriever_returns_glossary_for_rsi():
    project_root = Path(__file__).resolve().parents[1]
    retriever = RagRetriever(project_root=project_root)

    results = retriever.retrieve("What is RSI?", top_k=3)
    assert results
    assert any(r.chunk.source_path.replace(
        "\\", "/").endswith("data/corpus/glossary.md") for r in results)
    assert any("RSI" in r.chunk.text for r in results)
