from __future__ import annotations

from pydantic import BaseModel, Field

from omago_ai.agents.jargon_agent import JargonAgent
from omago_ai.agents.risk_agent import RiskAgent
from omago_ai.agents.synthesis_agent import SynthesisAgent
from omago_ai.llm.client import LLMClient
from omago_ai.llm.prompts import general_prompt
from omago_ai.orchestrator.parsing import parse_trade_request


class LeadResult(BaseModel):
    intent: str = Field(default="unknown")
    final_message: str
    risk: dict | None = None
    synthesis: dict | None = None
    market_used: dict | None = None
    jargon: dict | None = None
    sources: list[dict] | None = None


class LeadRouter:
    """Minimal placeholder.

    Next step: replace heuristics with real multi-agent orchestration and tool calls.
    """

    def handle(
        self,
        user_input: str,
        session,
        *,
        use_llm: bool = False,
        llm_client: LLMClient | None = None,
    ) -> LeadResult:
        text = user_input.strip().lower()

        if any(word in text for word in ["buy", "sell", "order", "trade"]):
            intent = "trade"
            trade = parse_trade_request(user_input)
            if trade is None:
                message = (
                    "I got this as a trade request, but I’m missing details.\n\n"
                    "Try: ‘buy ₹200 of TSLA’ or ‘sell 2 shares of AAPL’."
                )
                risk = None
                synthesis = None
            else:
                risk_agent = RiskAgent()
                synthesis_agent = SynthesisAgent()

                market = None
                ms = getattr(session, "market_snapshot", None)
                if isinstance(ms, dict):
                    market = ms.get(trade.ticker)

                risk_result = risk_agent.analyze(trade, market=market)
                synth_result = synthesis_agent.synthesize(
                    trade,
                    risk_result,
                    reading_level=getattr(session, "reading_level", "simple"),
                    use_llm=use_llm,
                    llm_client=llm_client,
                )

                risk = risk_result.model_dump()
                synthesis = synth_result.model_dump()

                steps = "\n".join([f"- {s}" for s in synth_result.steps])
                questions = "\n".join(
                    [f"- {q}" for q in synth_result.confirmation_questions])

                message = (
                    "Trade check (simple):\n\n"
                    f"{steps}\n\n"
                    "Before you confirm, answer:\n\n"
                    f"{questions}"
                )
        elif any(word in text for word in ["rsi", "ema", "macd", "support", "resistance", "chart"]):
            intent = "explain"
            agent = JargonAgent()
            jr = agent.explain(
                user_input,
                reading_level=getattr(session, "reading_level", "simple"),
                use_llm=use_llm,
                llm_client=llm_client,
            )
            message = jr.final_message
            sources = jr.sources
            jargon = jr.model_dump()
        else:
            intent = "general"
            message = (
                "Tell me what you want to do (example: 'buy ₹100 of AAPL' or 'explain RSI').\n\n"
                "I’ll respond in simple language and list clear next actions."
            )
            if use_llm and llm_client is not None:
                try:
                    system, user = general_prompt(
                        user_input, reading_level=getattr(
                            session, "reading_level", "simple")
                    )
                    llm_text = llm_client.complete(system=system, user=user)
                    if llm_text:
                        message = llm_text
                except Exception:
                    pass

        if getattr(session, "reading_level", "simple") == "simple":
            message = message.replace("understood", "got")

        return LeadResult(
            intent=intent,
            final_message=message,
            risk=(risk if intent == "trade" else None),
            synthesis=(synthesis if intent == "trade" else None),
            market_used=(market.model_dump() if (
                intent == "trade" and market is not None) else None),
            sources=(sources if intent == "explain" else None),
            jargon=(jargon if intent == "explain" else None),
        )
