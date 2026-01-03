from __future__ import annotations

from pydantic import BaseModel

from omago_ai.agents.risk_agent import RiskResult
from omago_ai.llm.client import LLMClient
from omago_ai.llm.prompts import synthesis_prompt
from omago_ai.llm.utils import extract_json_object
from omago_ai.orchestrator.parsing import TradeRequest


class SynthesisResult(BaseModel):
    steps: list[str]
    confirmation_questions: list[str]


class SynthesisAgent:
    """Converts analysis into accessible action steps.

    For hackathon: deterministic, short sentences, and explicit actions.
    """

    def synthesize(
        self,
        trade: TradeRequest,
        risk: RiskResult,
        *,
        reading_level: str = "simple",
        use_llm: bool = False,
        llm_client: LLMClient | None = None,
    ) -> SynthesisResult:
        amount_part = ""
        if trade.notional_usd is not None:
            amount_part = f"₹{trade.notional_usd:.0f} "
        elif trade.shares is not None:
            amount_part = f"{trade.shares:g} shares "

        action = "Buy" if trade.side == "buy" else "Sell"
        headline = f"{action} {amount_part}of {trade.ticker}".replace(
            "  ", " ").strip()

        steps: list[str] = []
        steps.append(f"Goal: {headline}.")
        steps.append(
            f"Risk level: {risk.risk_level} (score {risk.risk_score}/100).")

        if reading_level == "simple":
            steps.append("Read the risk reasons below.")
            for r in risk.reasons[:3]:
                steps.append(f"Why: {r}")
            steps.append("Pick one safer option:")
            for s in risk.safer_actions[:3]:
                steps.append(f"Option: {s}")
            steps.append(
                "If you still want to continue, confirm the details and place the order.")
        else:
            steps.extend([f"Reason: {r}" for r in risk.reasons[:3]])
            steps.extend(
                [f"Safer action: {s}" for s in risk.safer_actions[:3]])

        questions: list[str] = []
        questions.append("Is the ticker correct?")
        if trade.notional_usd is not None:
            questions.append("Is the rupee amount correct?")
        if trade.shares is not None:
            questions.append("Is the share quantity correct?")
        questions.append("Do you understand you could lose money?")

        # Optional LLM enhancement: produce a cleaner checklist.
        if use_llm and llm_client is not None:
            try:
                system, user = synthesis_prompt(
                    reading_level=reading_level,
                    trade_summary=headline,
                    risk_level=risk.risk_level,
                    risk_score=risk.risk_score,
                    reasons=risk.reasons[:5],
                    safer_actions=risk.safer_actions[:5],
                )
                text = llm_client.complete(system=system, user=user)
                obj = extract_json_object(text)
                llm_steps = obj.get("steps")
                llm_qs = obj.get("confirmation_questions")
                if isinstance(llm_steps, list) and isinstance(llm_qs, list):
                    steps = [str(s).strip()
                             for s in llm_steps if str(s).strip()][:12]
                    questions = [str(q).strip()
                                 for q in llm_qs if str(q).strip()][:12]
            except Exception:
                # Fall back to deterministic output on any LLM or parsing failure.
                pass

        return SynthesisResult(steps=steps, confirmation_questions=questions)
