from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from omago_ai.agents.chart_agent import ChartAgent
from omago_ai.agents.jargon_agent import JargonAgent
from omago_ai.orchestrator.router import LeadRouter
from omago_ai.orchestrator.state import SessionState
from omago_ai.market import MarketSimulator, SimulatorConfig, build_market_snapshot, default_universe
from omago_ai.market.utils import ticks_to_frame
from omago_ai.llm import get_openai_client_from_env


st.set_page_config(page_title="OmagoAI Prototype", layout="wide")

st.title("OmagoAI – Accessible Trading Companion (Prototype)")
st.caption("Education-only prototype. Not financial advice.")

with st.sidebar:
    st.subheader("User Preferences")
    reading_level = st.selectbox("Reading level", options=[
                                 "simple", "standard"], index=0)
    st.session_state["reading_level"] = reading_level

tab_assistant, tab_market = st.tabs(["Assistant", "Market Simulation"])

with tab_assistant:
    st.divider()
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.subheader("Ask a question or describe an action")
        user_input = st.text_area(
            "Example: 'I want to buy $200 of TSLA' or 'What does RSI mean?'",
            height=120,
        )

        btn_col_a, btn_col_b = st.columns([1, 1])

        with btn_col_a:
            send_offline = st.button("Send", type="primary")

        with btn_col_b:
            send_llm = st.button("Send (LLM)")

        if send_offline or send_llm:
            if not user_input.strip():
                st.warning("Type something first.")
            else:
                router = LeadRouter()
                market_ticks = st.session_state.get("market_ticks", [])
                session = SessionState(
                    reading_level=st.session_state.get(
                        "reading_level", "simple"),
                    market_snapshot=build_market_snapshot(
                        market_ticks) if market_ticks else None,
                )

                llm_client = None
                use_llm = bool(send_llm)
                if use_llm:
                    try:
                        llm_client = get_openai_client_from_env()
                    except Exception as e:
                        st.error(
                            "LLM mode is not configured yet. Install `pip install -e .[llm]` and set OPENAI_API_KEY in .env.\n\n"
                            f"Details: {e}"
                        )
                        use_llm = False

                result = router.handle(
                    user_input=user_input,
                    session=session,
                    use_llm=use_llm,
                    llm_client=llm_client,
                )
                st.session_state["last_result"] = result

        if "last_result" in st.session_state:
            st.divider()
            st.subheader("Response")
            st.write(st.session_state["last_result"].final_message)

            if getattr(st.session_state["last_result"], "sources", None):
                with st.expander("Sources"):
                    st.json(st.session_state["last_result"].sources)

            if getattr(st.session_state["last_result"], "market_used", None):
                with st.expander("Market context used (debug)"):
                    mu = st.session_state["last_result"].market_used
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Ticker", mu.get("ticker", ""))
                    with c2:
                        lp = mu.get("last_price")
                        st.metric(
                            "Last price", f"{float(lp):.2f}" if lp is not None else "n/a")
                    with c3:
                        v = mu.get("vol_per_sqrt_day_est")
                        st.metric(
                            "Vol (est)", f"{float(v):.4f}" if v is not None else "n/a")

                    st.write(
                        {
                            "recent_event_count": mu.get("recent_event_count"),
                            "last_event_headline": mu.get("last_event_headline"),
                            "last_ts": mu.get("last_ts"),
                        }
                    )

                    st.caption("Raw")
                    st.json(mu)

    with col2:
        st.subheader("What’s happening")
        st.write(
            "This UI is wired to a **Lead Router** that will later orchestrate Risk / Synthesis / Jargon agents."
        )

        st.divider()
        st.subheader("Glossary search")
        q = st.text_input(
            "Search a term", placeholder="RSI, EMA, support, resistance…")
        if st.button("Search", key="glossary_search"):
            if not q.strip():
                st.warning("Type a term first.")
            else:
                agent = JargonAgent()
                jr = agent.explain(q, reading_level=st.session_state.get(
                    "reading_level", "simple"))
                st.session_state["glossary_result"] = jr

        if "glossary_result" in st.session_state:
            jr = st.session_state["glossary_result"]
            st.write(jr.final_message)
            if jr.sources:
                with st.expander("Sources", expanded=False):
                    st.json(jr.sources)

        if "last_result" in st.session_state:
            st.caption("Debug")
            st.json(st.session_state["last_result"].model_dump())

with tab_market:
    st.divider()
    st.subheader("Market Simulation")
    st.caption("Simulated data for demo only. Not financial advice.")

    def _company_catalog() -> list[dict[str, Any]]:
        # Minimal static metadata for the demo UI.
        # If you later add real fundamentals, swap these placeholders out.
        return [
            {
                "ticker": "AAPL",
                "name": "Apple",
                "sector": "Technology",
                "description": "Consumer electronics and services.",
                "fundamentals": {
                    "Market Cap": "$2.9T",
                    "P/E": "30.1",
                    "EPS": "$6.43",
                    "Dividend Yield": "0.5%",
                    "52W High": "$260.0",
                    "52W Low": "$164.0",
                    "Beta": "1.2",
                },
            },
            {
                "ticker": "MSFT",
                "name": "Microsoft",
                "sector": "Technology",
                "description": "Cloud, software, and productivity tools.",
                "fundamentals": {
                    "Market Cap": "$3.1T",
                    "P/E": "35.2",
                    "EPS": "$11.2",
                    "Dividend Yield": "0.7%",
                    "52W High": "$460.0",
                    "52W Low": "$310.0",
                    "Beta": "0.9",
                },
            },
            {
                "ticker": "GOOG",
                "name": "Alphabet",
                "sector": "Technology",
                "description": "Search, ads, and cloud services.",
                "fundamentals": {
                    "Market Cap": "$2.1T",
                    "P/E": "25.0",
                    "EPS": "$6.10",
                    "Dividend Yield": "0.0%",
                    "52W High": "$200.0",
                    "52W Low": "$130.0",
                    "Beta": "1.1",
                },
            },
            {
                "ticker": "AMZN",
                "name": "Amazon",
                "sector": "Consumer Discretionary",
                "description": "E-commerce and cloud infrastructure.",
                "fundamentals": {
                    "Market Cap": "$1.9T",
                    "P/E": "55.0",
                    "EPS": "$3.00",
                    "Dividend Yield": "0.0%",
                    "52W High": "$210.0",
                    "52W Low": "$120.0",
                    "Beta": "1.3",
                },
            },
            {
                "ticker": "TSLA",
                "name": "Tesla",
                "sector": "Automotive",
                "description": "Electric vehicles and energy products.",
                "fundamentals": {
                    "Market Cap": "$0.8T",
                    "P/E": "60.0",
                    "EPS": "$4.30",
                    "Dividend Yield": "0.0%",
                    "52W High": "$320.0",
                    "52W Low": "$140.0",
                    "Beta": "1.9",
                },
            },
            {
                "ticker": "NVDA",
                "name": "NVIDIA",
                "sector": "Technology",
                "description": "GPUs for AI and accelerated computing.",
                "fundamentals": {
                    "Market Cap": "$2.4T",
                    "P/E": "65.0",
                    "EPS": "$2.10",
                    "Dividend Yield": "0.03%",
                    "52W High": "$150.0",
                    "52W Low": "$40.0",
                    "Beta": "1.6",
                },
            },
        ]

    def _ensure_market_sim() -> MarketSimulator:
        if "market_sim" not in st.session_state:
            st.session_state["market_sim"] = MarketSimulator(
                universe=default_universe(),
                config=SimulatorConfig(
                    seed=7, tick_seconds=1.0, event_probability_per_tick=0.02),
                start_ts=datetime.now(timezone.utc).replace(microsecond=0),
            )
        if "market_ticks" not in st.session_state:
            # Auto-generate initial ticks so the UI has data without showing simulator controls.
            st.session_state["market_ticks"] = st.session_state["market_sim"].run(
                240)
        if "market_orders" not in st.session_state:
            st.session_state["market_orders"] = []
        if "market_selected_ticker" not in st.session_state:
            st.session_state["market_selected_ticker"] = None
        return st.session_state["market_sim"]

    def _frame_from_ticks() -> pd.DataFrame:
        ticks = st.session_state.get("market_ticks", [])
        if not ticks:
            return pd.DataFrame(columns=["ts", "ticker", "price", "volume", "event_type", "headline", "impact"])
        return ticks_to_frame(ticks)

    def _latest_price(frame: pd.DataFrame, *, ticker: str, fallback: float) -> float:
        if frame.empty:
            return float(fallback)
        sub = frame[frame["ticker"] == ticker].tail(1)
        if sub.empty:
            return float(fallback)
        return float(sub.iloc[0]["price"])

    def _price_change(frame: pd.DataFrame, *, ticker: str) -> tuple[float | None, float | None]:
        if frame.empty:
            return None, None
        sub = frame[frame["ticker"] == ticker]
        if len(sub) < 2:
            return None, None
        first = float(sub.iloc[0]["price"])
        last = float(sub.iloc[-1]["price"])
        if first <= 0:
            return None, None
        abs_change = last - first
        pct_change = (abs_change / first) * 100.0
        return abs_change, pct_change

    def _explain_term(term: str) -> None:
        agent = JargonAgent()
        jr = agent.explain(
            term,
            reading_level=st.session_state.get("reading_level", "simple"),
            use_llm=False,
            llm_client=None,
        )
        st.session_state["market_term_explanation"] = {
            "term": term,
            "text": jr.final_message,
            "sources": jr.sources,
        }

    def _render_explanation_box() -> None:
        ex = st.session_state.get("market_term_explanation")
        if not ex:
            return
        st.divider()
        st.subheader(f"Meaning (simple): {ex.get('term', '')}")
        st.write(ex.get("text", ""))
        if ex.get("sources"):
            with st.expander("Sources", expanded=False):
                st.json(ex.get("sources"))

    def _render_chart_explanation() -> None:
        ex = st.session_state.get("market_chart_explanation")
        if not ex:
            return
        st.divider()
        st.subheader(f"Chart explanation (simple): {ex.get('ticker', '')}")
        st.write(ex.get("text", ""))

    sim = _ensure_market_sim()
    frame = _frame_from_ticks()
    specs = {s.ticker: s for s in default_universe()}
    catalog = _company_catalog()

    selected = st.session_state.get("market_selected_ticker")

    # --- Landing dashboard (company list) ---
    if not selected:
        st.subheader("Companies")

        # 3-column grid of company cards.
        cols = st.columns(3, gap="large")
        for i, c in enumerate(catalog):
            ticker = str(c["ticker"])
            spec = specs.get(ticker)
            start_price = float(spec.start_price) if spec else 100.0

            last_price = _latest_price(
                frame, ticker=ticker, fallback=start_price)
            chg_abs, chg_pct = _price_change(frame, ticker=ticker)

            with cols[i % 3]:
                st.markdown(f"#### {c['name']}")
                st.caption(f"{ticker} · {c['sector']}")
                st.write(c.get("description", ""))
                if chg_abs is None or chg_pct is None:
                    st.metric("Price", f"${last_price:,.2f}")
                else:
                    st.metric("Price", f"${last_price:,.2f}",
                              f"{chg_abs:+.2f} ({chg_pct:+.2f}%)")

                if st.button("Open", key=f"open_{ticker}", type="primary"):
                    st.session_state["market_selected_ticker"] = ticker
                    st.rerun()

        st.caption("Click a company to open its chart and fundamentals.")
        st.stop()

    # --- Company detail view ---
    ticker = str(selected)
    company = next((x for x in catalog if x["ticker"] == ticker), None)
    if company is None:
        st.session_state["market_selected_ticker"] = None
        st.rerun()

    top_left, top_right = st.columns([3, 1])
    with top_left:
        st.subheader(f"{company['name']} ({ticker})")
        st.caption(f"{company['sector']} · {company.get('description','')}")
    with top_right:
        if st.button("← Back to companies", key="back_to_companies"):
            st.session_state["market_selected_ticker"] = None
            st.rerun()

    spec = specs.get(ticker)
    start_price = float(spec.start_price) if spec else 100.0
    last_price = _latest_price(frame, ticker=ticker, fallback=start_price)
    chg_abs, chg_pct = _price_change(frame, ticker=ticker)

    header_a, header_b, header_c, header_d = st.columns(
        [1.2, 1.2, 1, 1], gap="large")
    with header_a:
        if chg_abs is None or chg_pct is None:
            st.metric("Current Price", f"${last_price:,.2f}")
        else:
            st.metric(
                "Current Price", f"${last_price:,.2f}", f"{chg_abs:+.2f} ({chg_pct:+.2f}%)")
    with header_b:
        # Show last timestamp if available.
        if not frame.empty:
            sub = frame[frame["ticker"] == ticker].tail(1)
            if not sub.empty:
                ts = sub.iloc[0]["ts"]
                st.metric("Last Update", str(ts))
            else:
                st.metric("Last Update", "n/a")
        else:
            st.metric("Last Update", "n/a")
    with header_c:
        if spec is not None:
            st.metric("Sim Vol", f"{float(spec.vol_per_sqrt_day):.3f}")
        else:
            st.metric("Sim Vol", "n/a")
    with header_d:
        st.metric("Status", "Simulated")

    chart_col, trade_col = st.columns([2.3, 1], gap="large")

    with chart_col:
        st.subheader("Price")
        lookback = st.selectbox(
            "Range",
            options=[60, 120, 240, 600],
            index=2,
            format_func=lambda x: {60: "1 min", 120: "2 min",
                                   240: "4 min", 600: "10 min"}.get(x, str(x)),
            key="market_lookback",
        )

        if frame.empty:
            st.info("No simulated data yet.")
        else:
            sub = frame[frame["ticker"] == ticker].tail(int(lookback))
            if sub.empty:
                st.info("No ticks for this ticker yet.")
            else:
                fig = px.line(sub, x="ts", y="price", title=None)
                fig.update_layout(margin=dict(
                    l=10, r=10, t=10, b=10), height=360)
                st.plotly_chart(fig, use_container_width=True)

                if st.button("Explain this chart", key=f"explain_chart_{ticker}"):
                    agent = ChartAgent()
                    ev = None
                    try:
                        ev = sub[sub["event_type"].notna(
                        )][["ts", "event_type", "headline", "impact"]]
                    except Exception:
                        ev = None

                    res = agent.explain(
                        ticker=ticker,
                        prices=sub[["ts", "price"]],
                        events=ev,
                        reading_level=st.session_state.get(
                            "reading_level", "simple"),
                    )
                    st.session_state["market_chart_explanation"] = {
                        "ticker": ticker,
                        "text": res.final_message,
                        "summary": res.summary,
                    }

    with trade_col:
        st.subheader("Buy / Sell")
        with st.form("trade_form", clear_on_submit=False):
            side = st.radio("Action", options=["Buy", "Sell"], horizontal=True)
            qty = st.number_input("Shares", min_value=0.0, value=1.0, step=1.0)
            est = float(qty) * float(last_price)
            st.write(f"Estimated value: **${est:,.2f}**")
            submitted = st.form_submit_button("Place order", type="primary")

        if submitted:
            order = {
                "ts": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "ticker": ticker,
                "side": side.lower(),
                "shares": float(qty),
                "estimated_price": float(last_price),
                "estimated_value": float(est),
            }
            st.session_state["market_orders"].append(order)
            st.success(
                f"Order saved (demo): {side} {qty:g} shares of {ticker}.")

        if st.session_state.get("market_orders"):
            with st.expander("Recent orders (demo)", expanded=False):
                st.dataframe(pd.DataFrame(st.session_state["market_orders"]).tail(
                    10), use_container_width=True)

    # Events section (kept, but positioned below the chart/trade panel)
    if not frame.empty:
        st.subheader("Recent news / events")
        events = frame[(frame["ticker"] == ticker) &
                       frame["event_type"].notna()].tail(10)
        if events.empty:
            st.write("No simulated events yet for this company.")
        else:
            st.dataframe(events[["ts", "event_type", "headline", "impact"]],
                         use_container_width=True, hide_index=True)

    # Fundamentals section (scroll-down content)
    st.subheader("Fundamentals")

    fundamentals: dict[str, str] = dict(company.get("fundamentals") or {})
    # Add a couple of live-ish items derived from the simulator.
    if not frame.empty:
        vol = None
        try:
            snapshot = build_market_snapshot(
                st.session_state.get("market_ticks", []))
            s = snapshot.get(ticker)
            if s is not None and s.vol_per_sqrt_day_est is not None:
                vol = float(s.vol_per_sqrt_day_est)
        except Exception:
            vol = None

        sub = frame[frame["ticker"] == ticker]
        if not sub.empty:
            fundamentals.setdefault(
                "Avg Volume (sim)", f"{float(sub['volume'].tail(240).mean()):,.0f}")
        if vol is not None:
            fundamentals.setdefault("Volatility (est, sim)", f"{vol:.3f}")

    # Present fundamentals in a compact grid, with clickable labels.
    items = list(fundamentals.items())
    grid = st.columns(4, gap="large")
    for i, (k, v) in enumerate(items):
        with grid[i % 4]:
            if st.button(str(k), key=f"fund_term_{ticker}_{i}"):
                _explain_term(str(k))
            st.write(f"**{v}**")

    # A simple financials table (placeholder values)
    st.subheader("Financials (sample)")
    fin_rows = [
        ("Revenue", "—"),
        ("Gross Profit", "—"),
        ("Net Income", "—"),
        ("Free Cash Flow", "—"),
    ]
    for i, (metric, value) in enumerate(fin_rows):
        c1, c2 = st.columns([1.2, 1], gap="large")
        with c1:
            if st.button(metric, key=f"fin_term_{ticker}_{i}"):
                _explain_term(metric)
        with c2:
            st.write(f"**{value}**")

    _render_explanation_box()
    _render_chart_explanation()
