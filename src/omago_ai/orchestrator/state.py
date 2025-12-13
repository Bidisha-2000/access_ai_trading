from __future__ import annotations

from pydantic import BaseModel, Field

from omago_ai.market.features import MarketTickerSnapshot


class SessionState(BaseModel):
    reading_level: str = Field(default="simple")
    market_snapshot: dict[str, MarketTickerSnapshot] | None = None
