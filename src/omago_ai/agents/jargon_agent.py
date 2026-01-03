# jargon_agent.py
# ---------------------------------------------
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel

try:
    from gtts import gTTS  # type: ignore
except Exception:  # pragma: no cover
    gTTS = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
    from langchain_core.prompts import PromptTemplate  # type: ignore
except Exception:  # pragma: no cover
    ChatGoogleGenerativeAI = None
    PromptTemplate = None

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
        api_key = os.getenv(api_key_env)

        self.llm = None
        if ChatGoogleGenerativeAI is not None and api_key:
            self.llm = ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                google_api_key=api_key,
            )

        self._simple_template = (
            "Explain this trading term in very simple language.\n\n"
            "Rules:\n"
            "- No jargon\n"
            "- Grade 5 reading level\n"
            "- 6-10 short sentences\n\n"
            'Term: "{term}"\n\n'
            "Return ONLY the explanation.\n"
        )

        self._narration_template = (
            "Explain this trading term like a calm teacher.\n\n"
            "Rules:\n"
            "- Friendly tone\n"
            "- Simple real-life example\n"
            "- Explain why it matters\n"
            "- 6–10 sentences\n"
            "- No jargon\n"
            "- Grade 5 reading level\n\n"
            'Term: "{term}"\n\n'
            "Return ONLY narration text.\n"
        )

        if PromptTemplate is not None:
            self.simple_prompt = PromptTemplate(
                input_variables=["term"],
                template=self._simple_template,
            )
            self.narration_prompt = PromptTemplate(
                input_variables=["term"],
                template=self._narration_template,
            )
        else:
            self.simple_prompt = None
            self.narration_prompt = None

    def _format_prompt(self, kind: str, *, term: str) -> str:
        template = self._simple_template if kind == "simple" else self._narration_template
        return template.format(term=term)

    # --------------------------------------------------
    # SAFE LLM CALL
    # --------------------------------------------------
    def _safe_invoke(self, prompt: str, retries: int = 1) -> str:
        for i in range(retries):
            try:
                if self.llm is not None:
                    res = self.llm.invoke(prompt)
                    if res and getattr(res, "content", None):
                        return str(res.content).strip()
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
        if self.simple_prompt is not None:
            prompt = self.simple_prompt.format(term=term)
        else:
            prompt = self._format_prompt("simple", term=term)
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
        if self.narration_prompt is not None:
            prompt = self.narration_prompt.format(term=term)
        else:
            prompt = self._format_prompt("narration", term=term)
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
            if gTTS is None:
                raise RuntimeError("gTTS is not installed")

            tts = gTTS(text=text, lang="en", slow=True)
            tts.save(out_path)
            return out_path
        except Exception:
            # Ensure file always exists
            with open(out_path, "wb") as f:
                f.write(b"")
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
