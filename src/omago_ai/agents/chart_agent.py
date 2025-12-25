from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from pydantic import BaseModel


class ChartResult(BaseModel):
    final_message: str
    summary: dict | None = None


class ChartAgent:
    """Explains a price chart in simple language.

    This is intentionally offline and deterministic for the prototype.
    """

    def explain(
        self,
        *,
        ticker: str,
        prices: pd.DataFrame,
        events: pd.DataFrame | None = None,
        reading_level: str = "simple",
    ) -> ChartResult:
        ticker = (ticker or "").strip().upper()

        if prices is None or prices.empty:
            return ChartResult(
                final_message=(
                    f"I can’t explain the {ticker} chart yet because there is no price data. "
                    "Generate some simulated ticks and try again."
                ),
                summary={"ticker": ticker, "points": 0},
            )

        if "price" not in prices.columns:
            return ChartResult(
                final_message="I can’t explain this chart because I’m missing the price series.",
                summary={"ticker": ticker, "points": int(len(prices))},
            )

        s = prices[[c for c in ["ts", "price"]
                    if c in prices.columns]].dropna()
        if s.empty:
            return ChartResult(
                final_message="I can’t explain this chart because the series is empty.",
                summary={"ticker": ticker, "points": 0},
            )

        # Ensure time ordering.
        if "ts" in s.columns:
            s = s.sort_values("ts")

        px = s["price"].astype(float).to_numpy()
        px = px[np.isfinite(px)]
        if px.size < 2:
            return ChartResult(
                final_message=(
                    f"The {ticker} chart only has one data point so far. "
                    "Add more ticks and try again."
                ),
                summary={"ticker": ticker, "points": int(px.size)},
            )

        start_price = float(px[0])
        end_price = float(px[-1])
        low = float(np.min(px))
        high = float(np.max(px))

        # Basic movement.
        pct_change = None
        if start_price > 0:
            pct_change = (end_price - start_price) / start_price * 100.0

        swing_pct = None
        if start_price > 0:
            swing_pct = (high - low) / start_price * 100.0

        # Trend: simple start vs end.
        trend = "flat"
        if pct_change is not None:
            if pct_change > 0.3:
                trend = "up"
            elif pct_change < -0.3:
                trend = "down"

        # Simple volatility estimate from log returns.
        vol = None
        if np.all(px > 0) and px.size >= 8:
            lr = np.diff(np.log(px))
            if lr.size > 0 and np.all(np.isfinite(lr)):
                vol = float(np.std(lr))

        # Time range.
        start_ts: datetime | None = None
        end_ts: datetime | None = None
        if "ts" in s.columns:
            try:
                start_ts = pd.to_datetime(s.iloc[0]["ts"]).to_pydatetime()
                end_ts = pd.to_datetime(s.iloc[-1]["ts"]).to_pydatetime()
            except Exception:
                start_ts = None
                end_ts = None

        # Events summary.
        event_count = 0
        last_event = None
        if events is not None and not events.empty:
            event_count = int(len(events))
            try:
                last = events.tail(1).iloc[0]
                last_event = {
                    "event_type": (str(last.get("event_type")) if last.get("event_type") is not None else None),
                    "headline": (str(last.get("headline")) if last.get("headline") is not None else None),
                }
            except Exception:
                last_event = None

        parts: list[str] = []
        if reading_level == "simple":
            parts.append("Here’s what this chart is showing (simple):")
        else:
            parts.append("Chart summary:")

        if start_ts and end_ts:
            parts.append(f"- Time window: {start_ts} to {end_ts}.")
        parts.append(f"- Start price: ${start_price:,.2f}.")
        parts.append(f"- End price: ${end_price:,.2f}.")

        if pct_change is not None:
            parts.append(f"- Change: {pct_change:+.2f}%.")

        parts.append(f"- Range: low ${low:,.2f}, high ${high:,.2f}.")
        if swing_pct is not None:
            parts.append(f"- Biggest swing vs start: ~{swing_pct:.2f}%.")

        if trend == "up":
            parts.append("- Overall trend: moving up.")
        elif trend == "down":
            parts.append("- Overall trend: moving down.")
        else:
            parts.append("- Overall trend: roughly flat.")

        if vol is not None:
            # Vol is unitless per-tick log-return std.
            if vol >= 0.02:
                parts.append(
                    "- The line is jumpy (bigger tick-to-tick moves).")
            elif vol >= 0.01:
                parts.append("- The line has some movement (medium wiggles).")
            else:
                parts.append("- The line is relatively smooth (small moves).")

        if event_count > 0:
            parts.append(f"- Simulated events in this window: {event_count}.")
            if last_event and last_event.get("headline"):
                parts.append(f"- Latest event: {last_event.get('headline')}")

        parts.append(
            "Note: this is simulated demo data, not real market data.")

        summary = {
            "ticker": ticker,
            "points": int(len(s)),
            "start_price": start_price,
            "end_price": end_price,
            "pct_change": pct_change,
            "low": low,
            "high": high,
            "trend": trend,
            "event_count": event_count,
        }

        return ChartResult(final_message="\n".join(parts), summary=summary)
