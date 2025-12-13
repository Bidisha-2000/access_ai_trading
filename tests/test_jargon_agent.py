from __future__ import annotations

from pathlib import Path

from omago_ai.agents.jargon_agent import JargonAgent
from omago_ai.rag.retriever import RagRetriever


def test_jargon_agent_explains_rsi_with_sources():
    root = Path(__file__).resolve().parents[1]
    agent = JargonAgent(retriever=RagRetriever(project_root=root))

    result = agent.explain("What is RSI?", reading_level="simple")
    assert result.matched is True
    assert "RSI" in result.final_message
    assert result.sources
    assert any(s["source"].replace(
        "\\", "/").endswith("data/corpus/glossary.md") for s in result.sources)
