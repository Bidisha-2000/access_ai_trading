from __future__ import annotations

from pydantic import BaseModel, Field

from omago_ai.market.features import MarketTickerSnapshot
from omago_ai.market.universe import default_universe
from omago_ai.orchestrator.parsing import TradeRequest


class RiskResult(BaseModel):
    risk_score: int = Field(ge=0, le=100)
    risk_level: str
    reasons: list[str]
    safer_actions: list[str]


class RiskAgent:
    """Rules-first risk analysis.

    For hackathon: fast, deterministic, and explainable.
    Later you can replace/augment this with an LLM risk agent.
    """

    def analyze(self, trade: TradeRequest, *, market: MarketTickerSnapshot | None = None) -> RiskResult:
        specs = {s.ticker: s for s in default_universe()}
        spec = specs.get(trade.ticker)

        reasons: list[str] = []
        safer: list[str] = []

        # Base score from volatility.
        # Prefer live simulated volatility (derived from recent ticks) when available.
        vol = float(spec.vol_per_sqrt_day) if spec else 0.02
        if market is not None and market.vol_per_sqrt_day_est is not None:
            vol = float(market.vol_per_sqrt_day_est)
            reasons.append(
                "Based on recent price swings in the live simulator.")
            if market.recent_event_count > 0:
                reasons.append(
                    "Recent simulated news/earnings events can increase uncertainty.")
                safer.append(
                    "Be extra careful when the price is reacting to news.")
        if vol >= 0.032:
            score = 78
            level = "high"
            reasons.append(
                "This stock is simulated as more volatile (bigger price swings).")
            safer.append("Consider a smaller amount to start.")
            safer.append(
                "Consider using a stop-loss / limit order (if available).")
        elif vol >= 0.022:
            score = 55
            level = "medium"
            reasons.append("This stock can move noticeably up and down.")
            safer.append("Consider splitting into 2 smaller buys/sells.")
        else:
            score = 30
            level = "low"
            reasons.append("This stock is simulated as less volatile.")

        # Size-based adjustment (very rough).
        if trade.notional_usd is not None:
            if trade.notional_usd >= 1000:
                score = min(100, score + 15)
                reasons.append("The dollar amount is relatively large.")
                safer.append("Double-check the amount before confirming.")
            elif trade.notional_usd >= 300:
                score = min(100, score + 5)
                reasons.append("The dollar amount is moderate.")

        # Ensure at least one actionable item.
        if not safer:
            safer.append(
                "Review your plan and confirm you understand the risks.")

        return RiskResult(risk_score=int(score), risk_level=level, reasons=reasons, safer_actions=safer)
