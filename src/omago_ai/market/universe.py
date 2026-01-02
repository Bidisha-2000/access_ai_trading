from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TickerSpec:
    ticker: str
    start_price: float
    drift_per_day: float
    vol_per_sqrt_day: float


def default_universe() -> list[TickerSpec]:
    # Drift/vol are purely for simulation. NOT predictions.
    return [
        TickerSpec(
            ticker="TCS.NS",
            start_price=3800.0,
            drift_per_day=0.00025,
            vol_per_sqrt_day=0.012,
        ),
        TickerSpec(
            ticker="INFY.NS",
            start_price=1500.0,
            drift_per_day=0.00030,
            vol_per_sqrt_day=0.013,
        ),
        TickerSpec(
            ticker="RELIANCE.NS",
            start_price=2800.0,
            drift_per_day=0.00028,
            vol_per_sqrt_day=0.014,
        ),
        TickerSpec(
            ticker="HDFCBANK.NS",
            start_price=1600.0,
            drift_per_day=0.00022,
            vol_per_sqrt_day=0.011,
        ),
        TickerSpec(
            ticker="ICICIBANK.NS",
            start_price=1100.0,
            drift_per_day=0.00024,
            vol_per_sqrt_day=0.012,
        ),
        TickerSpec(
            ticker="ITC.NS",
            start_price=450.0,
            drift_per_day=0.00018,
            vol_per_sqrt_day=0.010,
        ),
    ]

