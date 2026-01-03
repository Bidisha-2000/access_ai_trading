from __future__ import annotations

import json
from pathlib import Path

from omago_ai.market.company_financials import (
    CACHE_PATH,
    fetch_company_financials_yfinance,
)


TICKERS = [
    "TCS.NS",
    "INFY.NS",
    "RELIANCE.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "ITC.NS",
]


def main() -> int:
    out: dict[str, dict] = {}
    for t in TICKERS:
        print(f"Fetching {t}...")
        out[t] = fetch_company_financials_yfinance(t)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote: {CACHE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
