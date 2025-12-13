from __future__ import annotations

from datetime import datetime, timezone

from omago_ai.agents.risk_agent import RiskAgent
from omago_ai.market.features import build_market_snapshot
from omago_ai.market.simulator import MarketSimulator, SimulatorConfig
from omago_ai.market.universe import default_universe
from omago_ai.orchestrator.parsing import TradeRequest


def test_risk_agent_uses_live_market_snapshot_volatility():
    # Create a simulator with frequent events to increase variance slightly.
    sim = MarketSimulator(
        universe=default_universe(),
        config=SimulatorConfig(seed=42, tick_seconds=1.0,
                               event_probability_per_tick=0.25),
        start_ts=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )

    ticks = sim.run(180)  # 180 seconds * 6 tickers
    snapshot = build_market_snapshot(ticks, window_ticks=180)
    assert "TSLA" in snapshot

    trade = TradeRequest(side="buy", ticker="TSLA", notional_usd=200.0)
    result = RiskAgent().analyze(trade, market=snapshot["TSLA"])

    assert result.risk_score >= 0
    assert any("live simulator" in r.lower() for r in result.reasons)
