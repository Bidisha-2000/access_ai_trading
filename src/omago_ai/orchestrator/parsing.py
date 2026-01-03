from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TradeRequest:
    side: str  # "buy" | "sell"
    ticker: str
    notional_usd: float | None = None
    shares: float | None = None


_TICKER_RE = re.compile(r"\b([A-Z]{2,5})\b")
_MONEY_RE = re.compile(
    r"(?:₹\s*(\d+(?:\.\d+)?))"
    r"|(?:\b(\d+(?:\.\d+)?)\s*(?:inr|rupees?|rs)\b)"
    r"|(?:\$\s*(\d+(?:\.\d+)?))"
    r"|(?:\b(\d+(?:\.\d+)?)\s*(?:usd|dollars?)\b)",
    re.IGNORECASE,
)
_SHARES_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:shares?|sh)\b", re.IGNORECASE)


def parse_trade_request(text: str) -> TradeRequest | None:
    if not text or not text.strip():
        return None

    raw = text.strip()
    lower = raw.lower()

    if "buy" in lower:
        side = "buy"
    elif "sell" in lower:
        side = "sell"
    else:
        return None

    # Pick the first plausible ticker.
    ticker = None
    for m in _TICKER_RE.finditer(raw):
        candidate = m.group(1)
        # Filter out common non-tickers in user text.
        if candidate in {"USD", "INR", "RS"}:
            continue
        ticker = candidate
        break

    if not ticker:
        return None

    shares = None
    m_sh = _SHARES_RE.search(raw)
    if m_sh:
        try:
            shares = float(m_sh.group(1))
        except ValueError:
            shares = None

    notional = None
    m_money = _MONEY_RE.search(raw)
    if m_money:
        amount = m_money.group(1) or m_money.group(
            2) or m_money.group(3) or m_money.group(4)
        try:
            notional = float(amount)
        except ValueError:
            notional = None

    return TradeRequest(side=side, ticker=ticker, notional_usd=notional, shares=shares)
