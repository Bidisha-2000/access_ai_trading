from __future__ import annotations

import pandas as pd

from omago_ai.market.models import MarketTick


def ticks_to_frame(ticks: list[MarketTick]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts": t.ts,
                "ticker": t.ticker,
                "price": t.price,
                "volume": t.volume,
                "event_type": (t.event.event_type if t.event else None),
                "headline": (t.event.headline if t.event else None),
                "impact": (t.event.impact if t.event else None),
            }
            for t in ticks
        ]
    )
