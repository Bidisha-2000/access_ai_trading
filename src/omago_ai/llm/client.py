from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol


class LLMClient(Protocol):
    def complete(self, *, system: str, user: str) -> str: ...


@dataclass
class OpenAIClient:
    api_key: str
    model: str
    base_url: str | None = None
    timeout_seconds: float = 30.0

    _client: object | None = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            from openai import OpenAI
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "OpenAI SDK is not installed. Install optional deps: pip install -e .[llm]"
            ) from e

        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def complete(self, *, system: str, user: str) -> str:
        client = self._get_client()

        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            timeout=self.timeout_seconds,
        )

        content = resp.choices[0].message.content
        return (content or "").strip()


def get_openai_client_from_env() -> OpenAIClient:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL") or None

    return OpenAIClient(api_key=api_key, model=model, base_url=base_url)
