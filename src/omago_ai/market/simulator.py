from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from omago_ai.market.models import MarketEvent, MarketTick
from omago_ai.market.universe import TickerSpec


@dataclass
class SimulatorConfig:
    seed: int = 7
    tick_seconds: float = 1.0
    base_volume: int = 1200
    event_probability_per_tick: float = 0.02


class MarketSimulator:
    """A small deterministic-ish market simulator for demos.

    - Simulates per-ticker prices with geometric Brownian motion.
    - Adds occasional "events" as shock terms.

    This is for a hackathon demo only.
    """

    def __init__(
        self,
        universe: list[TickerSpec],
        config: SimulatorConfig | None = None,
        start_ts: datetime | None = None,
    ) -> None:
        self.universe = universe
        self.config = config or SimulatorConfig()
        self.rng = np.random.default_rng(self.config.seed)

        self.ts = start_ts or datetime.now(timezone.utc).replace(microsecond=0)
        self.prices: dict[str, float] = {
            t.ticker: float(t.start_price) for t in universe}

        # Precompute day scaling.
        self._dt_days = float(self.config.tick_seconds) / 86400.0
        self._sqrt_dt_days = float(np.sqrt(self._dt_days))

    def step(self) -> list[MarketTick]:
        self.ts = self.ts + timedelta(seconds=float(self.config.tick_seconds))
        ticks: list[MarketTick] = []

        for spec in self.universe:
            ticker = spec.ticker

            shock = float(self.rng.normal(
                loc=0.0, scale=spec.vol_per_sqrt_day * self._sqrt_dt_days))
            drift = float(spec.drift_per_day * self._dt_days)

            event = self._maybe_event(ticker)
            event_shock = float(event.impact) if event else 0.0

            log_ret = drift + shock + event_shock
            new_price = float(self.prices[ticker] * np.exp(log_ret))
            self.prices[ticker] = max(new_price, 0.01)

            volume = int(max(1, self.rng.lognormal(
                mean=np.log(self.config.base_volume), sigma=0.35)))

            ticks.append(
                MarketTick(ts=self.ts, ticker=ticker,
                           price=self.prices[ticker], volume=volume, event=event)
            )

        return ticks

    def run(self, n_ticks: int) -> list[MarketTick]:
        all_ticks: list[MarketTick] = []
        for _ in range(int(n_ticks)):
            all_ticks.extend(self.step())
        return all_ticks

    def _maybe_event(self, ticker: str) -> MarketEvent | None:
        if float(self.rng.random()) >= float(self.config.event_probability_per_tick):
            return None

        event_type = "earnings" if float(self.rng.random()) < 0.35 else "news"
        impact = float(self.rng.normal(loc=0.0, scale=0.015))

        if event_type == "earnings":
            headline = f"{ticker} earnings surprised the market"
        else:
            headline = f"{ticker} announces a business update"

        # Clamp shocks so the demo stays readable.
        impact = float(np.clip(impact, -0.05, 0.05))

        return MarketEvent(ts=self.ts, ticker=ticker, event_type=event_type, headline=headline, impact=impact)
