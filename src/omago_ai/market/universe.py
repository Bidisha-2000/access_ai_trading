from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TickerSpec:
    ticker: str
    start_price: float
    drift_per_day: float
    vol_per_sqrt_day: float


def default_universe() -> list[TickerSpec]:
    # Drift/vol here are purely for simulation. They are NOT predictions.
    return [
        TickerSpec("AAPL", 190.0, drift_per_day=0.0004,
                   vol_per_sqrt_day=0.020),
        TickerSpec("MSFT", 420.0, drift_per_day=0.0003,
                   vol_per_sqrt_day=0.018),
        TickerSpec("GOOG", 165.0, drift_per_day=0.00035,
                   vol_per_sqrt_day=0.019),
        TickerSpec("AMZN", 175.0, drift_per_day=0.00045,
                   vol_per_sqrt_day=0.022),
        TickerSpec("TSLA", 260.0, drift_per_day=0.00055,
                   vol_per_sqrt_day=0.035),
        TickerSpec("NVDA", 135.0, drift_per_day=0.00060,
                   vol_per_sqrt_day=0.032),
    ]
