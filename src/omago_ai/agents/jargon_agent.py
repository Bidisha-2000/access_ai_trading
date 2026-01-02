# jargon_agent.py
# ---------------------------------------------
from __future__ import annotations

import os
import time
import textwrap
from typing import Any, Dict, List, Optional
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from gtts import gTTS
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()


# --------------------------------------------------
# ROUTER-COMPATIBLE RESULT MODEL
# --------------------------------------------------
class JargonResult(BaseModel):
    final_message: str
    sources: Optional[List[Dict[str, Any]]] = None
    market_used: Optional[Dict[str, Any]] = None


# --------------------------------------------------
# JARGON AGENT
# --------------------------------------------------
class JargonAgent:
    def __init__(
        self,
        model: str = "gemini-2.5-flash-lite",
        temperature: float = 0.2,
        api_key_env: str = "MY_TOKEN",
    ):
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=os.getenv(api_key_env),
        )

        self.simple_prompt = PromptTemplate(
            input_variables=["term"],
            template="""
Explain this trading term in very simple language.

Rules:
- No jargon
- Grade 5 reading level
- 6-10 short sentences


Term: "{term}"

Return ONLY the explanation.
""",
        )

        self.narration_prompt = PromptTemplate(
            input_variables=["term"],
            template="""
Explain this trading term like a calm teacher.

Rules:
- Friendly tone
- Simple real-life example
- Explain why it matters
- 6–10 sentences
- No jargon
- Grade 5 reading level

Term: "{term}"

Return ONLY narration text.
""",
        )

    # --------------------------------------------------
    # SAFE LLM CALL
    # --------------------------------------------------
    def _safe_invoke(self, prompt: str, retries: int = 1) -> str:
        for i in range(retries):
            try:
                res = self.llm.invoke(prompt)
                if res and res.content:
                    return res.content.strip()
            except Exception:
                time.sleep(2 * (i + 1))

        return "This term helps people understand the stock market."

    # --------------------------------------------------
    # ROUTER-SAFE EXPLAIN
    # --------------------------------------------------
    def explain(
        self,
        term: str,
        reading_level: str = "simple",
        **kwargs,
    ) -> JargonResult:
        prompt = self.simple_prompt.format(term=term)
        text = self._safe_invoke(prompt)

        return JargonResult(
            final_message=text,
            sources=[{"type": "llm", "model": "gemini"}],
        )

    # --------------------------------------------------
    # NARRATION (KWARG SAFE)
    # --------------------------------------------------
    def narrate(
        self,
        term: str,
        reading_level: str = "simple",
        **kwargs,
    ) -> str:
        prompt = self.narration_prompt.format(term=term)
        return self._safe_invoke(prompt)

    # --------------------------------------------------
    # TEXT → SPEECH (BYTES, STREAMLIT-SAFE)
    # --------------------------------------------------
    def text_to_speech(
    self,
    text: str,
    out_path: str = "narration.mp3",
) -> str:
        try:
            tts = gTTS(text=text, lang="en", slow=True)
            tts.save(out_path)
            return out_path
        except Exception:
            # Ensure file always exists
            with open(out_path, "wb") as f:
                f.write(b"")
            return out_path


    # --------------------------------------------------
    # VISUAL CARD (MATPLOTLIB)
    # --------------------------------------------------
    def create_visual(
        self,
        term: str,
        explanation: Any,
        out_path: str = "jargon_card.png",
    ) -> str:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")

        ax.text(
            0.5,
            0.85,
            term,
            fontsize=22,
            ha="center",
            weight="bold",
        )

        if hasattr(explanation, "final_message"):
            text = explanation.final_message
        else:
            text = str(explanation)

        wrapped = "\n".join(textwrap.wrap(text, width=42))
        ax.text(
            0.5,
            0.5,
            wrapped,
            fontsize=14,
            ha="center",
            va="center",
        )

        y = np.random.normal(0, 0.4, 30).cumsum()
        y_norm = (y - y.min()) / (y.max() - y.min())
        ax.plot(
            np.linspace(0.1, 0.9, 30),
            y_norm * 0.2 + 0.15,
            linewidth=2,
        )

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

        return out_path

    # --------------------------------------------------
    # UI-ONLY HELPER (NOT USED BY ROUTER)
    # --------------------------------------------------
    def explain_with_audio(self, term: str) -> Dict[str, Any]:
        jr = self.explain(term)
        narration = self.narrate(term)
        audio_bytes = self.text_to_speech(narration)

        return {
            "term": term,
            "text": jr.final_message,
            "audio": audio_bytes,
            "sources": jr.sources,
        }
