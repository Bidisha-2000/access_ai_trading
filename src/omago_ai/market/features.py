from __future__ import annotations

from datetime import datetime
from statistics import median

import numpy as np
from pydantic import BaseModel, Field

from omago_ai.market.models import MarketTick


class MarketTickerSnapshot(BaseModel):
    ticker: str
    last_price: float
    last_ts: datetime

    # Estimated volatility in the same units as TickerSpec.vol_per_sqrt_day.
    vol_per_sqrt_day_est: float | None = None

    recent_event_count: int = Field(default=0, ge=0)
    last_event_headline: str | None = None


def build_market_snapshot(
    ticks: list[MarketTick],
    *,
    window_ticks: int = 240,
) -> dict[str, MarketTickerSnapshot]:
    """Build per-ticker summary stats from recent ticks.

    Designed for the UI session-state: small, fast, and serializable.
    """

    if not ticks:
        return {}

    by_ticker: dict[str, list[MarketTick]] = {}
    for t in ticks:
        by_ticker.setdefault(t.ticker, []).append(t)

    out: dict[str, MarketTickerSnapshot] = {}

    for ticker, series in by_ticker.items():
        series = sorted(series, key=lambda x: x.ts)[-int(window_ticks):]
        last = series[-1]

        # Estimate tick interval.
        if len(series) >= 3:
            diffs = [
                max(0.001, (series[i].ts - series[i - 1].ts).total_seconds())
                for i in range(1, len(series))
            ]
            tick_seconds = float(median(diffs))
        else:
            tick_seconds = 1.0

        prices = np.array([float(x.price) for x in series], dtype=float)
        vol_est = None
        if len(prices) >= 12 and np.all(prices > 0):
            logret = np.diff(np.log(prices))
            std = float(np.std(logret))
            dt_days = tick_seconds / 86400.0
            if dt_days > 0:
                vol_est = float(std / np.sqrt(dt_days))

        events = [x.event for x in series if x.event is not None]
        out[ticker] = MarketTickerSnapshot(
            ticker=ticker,
            last_price=float(last.price),
            last_ts=last.ts,
            vol_per_sqrt_day_est=vol_est,
            recent_event_count=len(events),
            last_event_headline=(events[-1].headline if events else None),
        )

    return out
