from __future__ import annotations

import re

from pydantic import BaseModel, Field

from omago_ai.rag import RagRetriever
from omago_ai.llm.client import LLMClient
from omago_ai.llm.prompts import jargon_prompt


class JargonResult(BaseModel):
    final_message: str
    sources: list[dict] | None = None
    matched: bool = Field(default=False)


class JargonAgent:
    """Explains trading jargon using local RAG.

    This is intentionally simple and offline-friendly for hackathon demos.
    """

    def __init__(self, *, retriever: RagRetriever | None = None) -> None:
        self.retriever = retriever or RagRetriever()

    def explain(
        self,
        user_question: str,
        *,
        reading_level: str = "simple",
        use_llm: bool = False,
        llm_client: LLMClient | None = None,
    ) -> JargonResult:
        retrieved = self.retriever.retrieve(user_question, top_k=4)
        sources = [
            {"source": r.chunk.source_path, "score": round(r.score, 4)}
            for r in retrieved
        ] or None

        if not retrieved:
            return JargonResult(
                matched=False,
                final_message=(
                    "I can explain that in simple words, but I don’t have a matching glossary entry yet.\n\n"
                    "Add a short definition into `data/corpus/` (like `data/corpus/glossary.md`) and try again."
                ),
                sources=None,
            )

        # Prefer a paragraph that actually mentions the user's term (e.g., "RSI"),
        # not just a document title.
        best = retrieved[0].chunk.text.strip()
        snippet = _pick_relevant_paragraph(best, user_question)

        if reading_level == "simple":
            header = "Here’s a simple explanation:"
            followup = "Want me to explain a chart/alert too? Paste it here."
        else:
            header = "Explanation:"
            followup = "If you share the chart/alert text, I can explain it step-by-step."

        final_message = f"{header}\n\n{snippet}\n\n{followup}"

        if use_llm and llm_client is not None:
            try:
                system, user = jargon_prompt(
                    user_question,
                    reading_level=reading_level,
                    retrieved_snippets=[r.chunk.text for r in retrieved],
                )
                llm_text = llm_client.complete(system=system, user=user)
                if llm_text:
                    final_message = llm_text
            except Exception:
                # If LLM fails, silently fall back to offline explanation.
                pass

        return JargonResult(matched=True, final_message=final_message, sources=sources)


def _pick_relevant_paragraph(text: str, question: str) -> str:
    # Split on blank lines, allowing whitespace on the blank line.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if not paragraphs:
        return text.strip()

    q = (question or "").strip()
    q_lower = q.lower()

    # Heuristic: find the first paragraph containing a strong keyword.
    # Works well for glossary questions like "What is RSI?".
    keywords: list[str] = []
    for token in q.replace("?", " ").replace(",", " ").split():
        t = token.strip().strip("\"'()[]{}:;.!\n\t")
        if 2 <= len(t) <= 12 and any(c.isalpha() for c in t):
            keywords.append(t)

    def _maybe_expand_heading(i: int) -> str:
        p = paragraphs[i]
        # If we matched a markdown heading, also include the next paragraph as the definition.
        if p.lstrip().startswith("#") and i + 1 < len(paragraphs):
            return f"{p}\n\n{paragraphs[i + 1]}"
        return p

    # Prefer uppercase acronym matches (RSI/EMA/MACD) if present.
    for k in keywords:
        if k.isupper() and k.lower() in q_lower:
            for i, p in enumerate(paragraphs):
                if k.lower() in p.lower():
                    return _maybe_expand_heading(i)

    # Otherwise, pick the first paragraph that overlaps any keyword.
    for k in keywords:
        for i, p in enumerate(paragraphs):
            if k.lower() in p.lower():
                return _maybe_expand_heading(i)

    # Fallback: first paragraph.
    return paragraphs[0]
