from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class MarketEvent(BaseModel):
    ts: datetime
    ticker: str
    event_type: str
    headline: str
    impact: float = Field(
        description="Additive log-return shock applied on this tick (e.g., 0.02 ~ +2%)."
    )


class MarketTick(BaseModel):
    ts: datetime
    ticker: str
    price: float
    volume: int
    event: MarketEvent | None = None
