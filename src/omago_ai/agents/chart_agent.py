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

        def _r(amount: float) -> str:
            # Rounded rupees for accessibility.
            return f"₹{amount:,.0f}"

        # Basic movement.
        pct_change = None
        if start_price > 0:
            pct_change = (end_price - start_price) / start_price * 100.0

        abs_change = end_price - start_price

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

        # A cognitively-accessible explanation: short, structured, minimal numbers.
        parts.append("What you are looking at:")
        parts.append("- Left to right is time.")
        parts.append("- Up and down is the price.")
        parts.append(
            "- If the line goes up, the price is going up. If it goes down, the price is going down.")

        if start_ts and end_ts:
            parts.append("")
            parts.append(f"Time shown: {start_ts} to {end_ts}.")

        parts.append("")
        if trend == "up":
            parts.append(
                "Simple result: The price ended higher than it started.")
        elif trend == "down":
            parts.append(
                "Simple result: The price ended lower than it started.")
        else:
            parts.append(
                "Simple result: The price ended close to where it started.")

        parts.append(f"Start (left side): about {_r(start_price)}.")
        parts.append(f"End (right side): about {_r(end_price)}.")

        # Keep the change as one easy number.
        if abs_change != 0 and np.isfinite(abs_change):
            direction = "up" if abs_change > 0 else "down"
            parts.append(
                f"That is {direction} by about {_r(abs(abs_change))}.")

        parts.append("")
        parts.append("Highs and lows in this time:")
        parts.append(f"- Highest point: about {_r(high)}.")
        parts.append(f"- Lowest point: about {_r(low)}.")

        if vol is not None:
            parts.append("")
            parts.append("How bumpy the line looks:")
            if vol >= 0.02:
                parts.append("- Very bumpy: lots of quick up/down moves.")
            elif vol >= 0.01:
                parts.append("- Some bumps: a few quick moves.")
            else:
                parts.append("- Mostly smooth: small moves.")

        if event_count > 0:
            parts.append("")
            parts.append("Possible reason for a sudden move:")
            parts.append(
                f"- There were {event_count} simulated event(s) in this time.")
            if last_event and last_event.get("headline"):
                parts.append(
                    f"- Latest event headline: {last_event.get('headline')}")

        parts.append("")
        parts.append(
            "Reminder: this is simulated demo data, not real market data.")

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
