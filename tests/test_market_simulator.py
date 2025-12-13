from __future__ import annotations

from datetime import datetime, timezone

from omago_ai.market.simulator import MarketSimulator, SimulatorConfig
from omago_ai.market.universe import default_universe


def test_market_simulator_deterministic_first_tick():
    sim = MarketSimulator(
        universe=default_universe(),
        config=SimulatorConfig(seed=123, tick_seconds=1.0,
                               event_probability_per_tick=0.0),
        start_ts=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )

    ticks = sim.step()
    assert len(ticks) == 6
    assert {t.ticker for t in ticks} == {
        "AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "NVDA"}

    # No events when probability is 0.
    assert all(t.event is None for t in ticks)

    # Sanity: prices remain positive and close to start.
    for t in ticks:
        assert t.price > 0
        assert t.volume > 0
