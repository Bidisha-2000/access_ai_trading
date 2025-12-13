from __future__ import annotations

from dataclasses import dataclass

from omago_ai.orchestrator.router import LeadRouter
from omago_ai.orchestrator.state import SessionState


@dataclass
class FakeLLM:
    response: str

    def complete(self, *, system: str, user: str) -> str:  # noqa: ARG002
        return self.response


def test_router_trade_llm_mode_uses_llm_synthesis():
    router = LeadRouter()
    fake = FakeLLM(
        '{"steps": ["Step A"], "confirmation_questions": ["Q1?"]}'
    )

    res = router.handle(
        "buy $200 of TSLA",
        session=SessionState(reading_level="simple"),
        use_llm=True,
        llm_client=fake,
    )

    assert res.intent == "trade"
    assert res.synthesis is not None
    assert "Step A" in res.final_message


def test_router_general_llm_mode_uses_llm_text():
    router = LeadRouter()
    fake = FakeLLM("Hello from LLM")

    res = router.handle(
        "hi",
        session=SessionState(reading_level="simple"),
        use_llm=True,
        llm_client=fake,
    )

    assert res.intent == "general"
    assert "Hello from LLM" in res.final_message
