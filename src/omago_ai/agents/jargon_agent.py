# jargon_agent.py
# ---------------------------------------------
# Requirements:
# pip install langchain langchain-google-genai matplotlib gtts python-dotenv

from __future__ import annotations

import os
import time
import textwrap
import numpy as np
import matplotlib.pyplot as plt
from gtts import gTTS
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Any, Dict, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()


# --------------------------------------------------
# Result object (router-compatible)
# --------------------------------------------------
@dataclass
class JargonResult:
    final_message: str
    sources: Optional[Dict[str, Any]] = None
    market_used: Optional[Dict[str, Any]] = None


# --------------------------------------------------
# Jargon Agent
# --------------------------------------------------
class JargonAgent:
    """
    JargonAgent:
    - Explains trading terms in simple language
    - Generates a visual card
    - Generates calm narration audio
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash-lite",  # 👈 more stable than 2.5
        temperature: float = 0.2,
        api_key_env: str = "MY_TOKEN",
    ):
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=os.getenv(api_key_env),
        )

        # -------- Prompts --------
        self.simple_prompt = PromptTemplate(
            input_variables=["term"],
            template="""
Explain this trading term in simple 1–2 sentences.

Rules:
- No jargon
- Grade 5 reading level
- Very short sentences

Term: "{term}"

Return ONLY the simple explanation.
"""
        )

        self.narration_prompt = PromptTemplate(
            input_variables=["term"],
            template="""
You are explaining a trading term to a beginner using storytelling.

Rules:
- Speak like a calm teacher or mentor
- Use a simple real-life example or short story
- Try to make it relatable with the stock market(make it simple)
- No jargon
- Short sentences
- Friendly and reassuring tone
- 7 - 15 sentences max
- Grade 5 reading level
- Explain WHY it matters

Term: "{term}"

Return ONLY the narration text.
"""
        )

    # --------------------------------------------------
    # SAFE LLM INVOKE (handles overload)
    # --------------------------------------------------
    def _safe_invoke(self, prompt: str, retries: int = 3) -> str:
        for i in range(retries):
            try:
                response = self.llm.invoke(prompt)
                text = response.content.strip()
                if text:
                    return text
            except Exception:
                time.sleep(2 * (i + 1))

        # Fallback if Gemini is overloaded
        return "This term explains how prices move in the market."

    # --------------------------------------------------
    # TEXT EXPLANATION (READING)
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
            sources={"type": "llm", "model": "gemini"},
        )

    # --------------------------------------------------
    # NARRATION TEXT (LISTENING) — FIXED
    # --------------------------------------------------
    def narrate(
        self,
        term: str,
        reading_level: str = "simple",
        **kwargs,
    ) -> str:
        prompt = self.narration_prompt.format(term=term)
        text = self._safe_invoke(prompt)

        # Absolute safety: narration must be non-empty
        if not text.strip():
            text = "This is a simple trading term used to understand the market."

        return text

    # --------------------------------------------------
    # VISUAL CARD
    # --------------------------------------------------
    def create_visual(
    self,
    term: str,
    explanation: JargonResult | str,
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

        if isinstance(explanation, JargonResult):
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
    # TEXT → SPEECH (SAFE)
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
            # absolute fallback
            with open(out_path, "wb") as f:
                f.write(b"")
            return out_path
