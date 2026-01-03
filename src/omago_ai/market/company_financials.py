from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    # .../src/omago_ai/market/company_financials.py -> .../<root>
    return Path(__file__).resolve().parents[3]


CACHE_PATH = _project_root() / "data" / "company_financials.json"


@dataclass(frozen=True)
class CompanyFinancials:
    ticker: str
    currency: str | None
    as_of: str | None
    rows: dict[str, str]


def load_company_financials_cache(path: Path = CACHE_PATH) -> dict[str, CompanyFinancials]:
    if not path.exists():
        return {}

    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, CompanyFinancials] = {}
    for ticker, payload in (raw or {}).items():
        if not isinstance(payload, dict):
            continue
        rows = payload.get("rows") or {}
        if not isinstance(rows, dict):
            rows = {}
        out[str(ticker)] = CompanyFinancials(
            ticker=str(ticker),
            currency=payload.get("currency"),
            as_of=payload.get("as_of"),
            rows={str(k): str(v) for k, v in rows.items()},
        )
    return out


def format_money_inr(value: float | int | None) -> str:
    if value is None:
        return "—"

    try:
        v = float(value)
    except Exception:
        return "—"

    av = abs(v)
    # Prefer crore for India readability.
    if av >= 1e7:
        return f"₹{v / 1e7:,.0f} Cr"
    if av >= 1e5:
        return f"₹{v:,.0f}"
    return f"₹{v:,.2f}"


def _latest_col_value(df: Any, labels: list[str]) -> float | None:
    """Extract the most recent (first) column value for any matching row label."""
    if df is None:
        return None

    try:
        if getattr(df, "empty", False):
            return None
        cols = list(getattr(df, "columns", []))
        if not cols:
            return None
        col0 = cols[0]
        idx = getattr(df, "index", [])
        for label in labels:
            if label in idx:
                val = df.loc[label, col0]
                try:
                    if getattr(val, "item", None):
                        val = val.item()
                except Exception:
                    pass
                try:
                    return float(val)
                except Exception:
                    return None
    except Exception:
        return None

    return None


def fetch_company_financials_yfinance(ticker: str) -> dict[str, Any]:
    """Fetch a small set of financial statement metrics via yfinance.

    Returns a dict suitable to store in CACHE_PATH.
    """
    import yfinance as yf  # local import so UI can run without yfinance

    t = yf.Ticker(ticker)

    # Annual statements (most recent period is typically first column)
    income = getattr(t, "financials", None)
    cashflow = getattr(t, "cashflow", None)

    revenue = _latest_col_value(income, ["Total Revenue", "TotalRevenue"])
    gross_profit = _latest_col_value(income, ["Gross Profit", "GrossProfit"])
    net_income = _latest_col_value(income, ["Net Income", "NetIncome"])

    free_cf = _latest_col_value(cashflow, ["Free Cash Flow", "FreeCashFlow"])
    if free_cf is None:
        op_cf = _latest_col_value(
            cashflow, ["Total Cash From Operating Activities", "Operating Cash Flow"])
        capex = _latest_col_value(
            cashflow, ["Capital Expenditures", "CapitalExpenditures"])
        if op_cf is not None and capex is not None:
            # capex is usually negative in statements
            free_cf = op_cf + capex

    currency = None
    try:
        info = getattr(t, "fast_info", None)
        if info and isinstance(info, dict):
            currency = info.get("currency")
    except Exception:
        currency = None

    rows = {
        "Revenue": format_money_inr(revenue),
        "Gross Profit": format_money_inr(gross_profit),
        "Net Income": format_money_inr(net_income),
        "Free Cash Flow": format_money_inr(free_cf),
    }

    return {
        "as_of": date.today().isoformat(),
        "currency": currency or "INR",
        "rows": rows,
    }
