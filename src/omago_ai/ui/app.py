from __future__ import annotations
import json
from datetime import datetime, timezone
import os
from typing import Any
import pandas as pd
import plotly.express as px
import streamlit as st
from omago_ai.agents.chart_agent import ChartAgent
from omago_ai.agents.jargon_agent import JargonAgent
from omago_ai.orchestrator.router import LeadRouter
from omago_ai.agents.risk_agent import RiskAgent
from omago_ai.orchestrator.parsing import TradeRequest
from omago_ai.orchestrator.state import SessionState
from omago_ai.market import MarketSimulator, SimulatorConfig, build_market_snapshot, default_universe
from omago_ai.market.utils import ticks_to_frame
from omago_ai.llm import get_openai_client_from_env
import re
from omago_ai.agents.news_agent import NewsAgent


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

FUNDAMENTAL_AUDIO = {
    "Market Cap": "audio_market_cap.mp3",
    "P/E": "audio_pe.mp3",
    "EPS": "audio_eps.mp3",
    "Dividend Yield": "audio_dividend_yield.mp3",
    "52W High": "audio_52w_high.mp3",
    "52W Low": "audio_52w_low.mp3",
    "Beta": "audio_beta.mp3",
    "Volatility (est, sim)": "audio_volatility_est_sim.mp3",
    "Avg Volume (sim)": "audio_avg_volume_sim.mp3",
}


if "show_jargon" not in st.session_state:
    st.session_state["show_jargon"] = False



def extract_json(text: str) -> dict:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}
    try:
        return json.loads(match.group())
    except Exception:
        return {}

def play_fundamental_audio(label: str):
    audio_rel = FUNDAMENTAL_AUDIO.get(label)
    if not audio_rel:
        return

    audio_abs = os.path.join(BASE_DIR,"assets",audio_rel)

    if not os.path.exists(audio_abs):
        st.error(f"Audio file missing: {audio_abs}")
        return
    
    if os.path.exists(audio_abs):
        with open(audio_abs, "rb") as f:
            st.session_state["fundamental_audio_bytes"] = f.read()
            st.session_state["fundamental_audio_label"] = label



def _safe_filename(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)   # remove / % . etc
    text = re.sub(r"\s+", "_", text)       # spaces → _
    return text

def _explain_term(term: str, *, with_audio: bool = True) -> None:
    agent = JargonAgent()

    jr = agent.explain(term)

    img_path = f"card_{term.replace(' ', '_').lower()}.png"
    agent.create_visual(term, jr, img_path)

    audio_path = None
    if with_audio:
        narration = agent.narrate(term)
        audio_path = agent.text_to_speech(
            narration,
            f"narration_{term.replace(' ', '_').lower()}.mp3"
        )

    st.session_state["glossary_result"] = {
        "term": term,
        "image": img_path,
        "audio": audio_path,
        "sources": jr.sources,
    }

# ---------------- EVENT DATA (SAFE, NO UI) ----------------
def get_latest_events(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    events_all = frame[
        (frame["ticker"] == ticker) & frame["event_type"].notna()
    ]

    if events_all.empty:
        return pd.DataFrame()

    latest_news = (
        events_all[events_all["event_type"] == "news"]
        .sort_values("ts", ascending=False)
        .head(1)
    )

    latest_earnings = (
        events_all[events_all["event_type"] == "earnings"]
        .sort_values("ts", ascending=False)
        .head(1)
    )

    return pd.concat([latest_news, latest_earnings]).sort_values("ts")

def render_jargon_card():
    if "glossary_result" not in st.session_state:
        return

    gr = st.session_state["glossary_result"]

    st.divider()
    st.subheader(f"📘 {gr['term']} — meaning")

    with st.container(border=True):
        st.image(gr["image"], use_column_width=True)

        if gr.get("audio"):
            st.audio(gr["audio"], format="audio/mp3")

        if gr.get("sources"):
            with st.expander("Sources"):
                st.json(gr["sources"])




st.set_page_config(page_title="OmagoAI Prototype", layout="wide")

#Call the risk agent 
risk_agent = RiskAgent()
#Call news agent
news_agent = NewsAgent()

st.title("OmagoAI – Orchestrated Multi-Agent Guardian for Open-trading ")
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
                
                # ---------------- RISK CHECK (Assistant) ----------------
                try:
                    if hasattr(result, "trade_request") and result.trade_request:
                        trade_req = result.trade_request

                        risk = risk_agent.analyze(trade_req)

                        st.session_state["assistant_risk"] = risk
                except Exception as e:
                    st.session_state["assistant_risk_error"] = str(e)


        if "last_result" in st.session_state:
            st.divider()
            st.subheader("Response")
            st.write(st.session_state["last_result"].final_message)

            # ---------------- RISK OUTPUT (Assistant) ----------------
            if "assistant_risk" in st.session_state:
                r = st.session_state["assistant_risk"]

                st.subheader("⚠️ Risk Check")
                st.metric("Final Risk Score", f"{r.final_risk_score}/100", r.risk_level.upper())
                st.write("Reason:", r.reasons[0])


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
        q = st.text_input("Search a term", placeholder="RSI, EMA, support, resistance…")

        if st.button("Search", key="glossary_search"):
            if not q.strip():
                st.warning("Type a term first.")
            else:
                _explain_term(q)

        # -------- CARD POP-OUT RENDER --------
        if "glossary_result" in st.session_state:
            gr = st.session_state["glossary_result"]

            st.divider()
            st.subheader("📘 Term explanation")

            with st.container(border=True):
                st.image(gr["image"], use_column_width=True)

                if gr.get("audio"):
                    st.audio(gr["audio"])

                if gr.get("sources"):
                    with st.expander("Sources"):
                        st.json(gr["sources"])




        #if "glossary_result" in st.session_state:
        # jr = st.session_state["glossary_result"]
        #st.write(jr.final_message)
            #if jr.sources:
                #with st.expander("Sources", expanded=False):
                    #st.json(jr.sources)

        if "last_result" in st.session_state:
            st.caption("Debug")
            st.json(st.session_state["last_result"].model_dump())

with tab_market:
    st.divider()
    st.subheader("Market Simulation")
    st.caption("Simulated data for demo only. Not financial advice.")

    def _company_catalog() -> list[dict[str, Any]]:
        return [
        {
            "ticker": "TCS.NS",
            "name": "Tata Consultancy Services",
            "sector": "IT Services",
            "description": "India’s largest IT services company.",
            "fundamentals": {
                "Market Cap": "$14.5T",
                "P/E": "28.0",
                "EPS": "$80",
                "Dividend Yield": "1.2%",
                "52W High": "$4,260",
                "52W Low": "$3,200",
                "Beta": "0.9",
            },
        },
        {
            "ticker": "INFY.NS",
            "name": "Infosys",
            "sector": "IT Services",
            "description": "Global consulting and IT services firm.",
            "fundamentals": {
                "Market Cap": "$6.5T",
                "P/E": "25.0",
                "EPS": "$65",
                "Dividend Yield": "1.8%",
                "52W High": "$1,980",
                "52W Low": "$1,350",
                "Beta": "0.8",
            },
        },
        {
            "ticker": "RELIANCE.NS",
            "name": "Reliance Industries",
            "sector": "Conglomerate",
            "description": "Energy, telecom, retail, and digital services.",
            "fundamentals": {
                "Market Cap": "$18.8T",
                "P/E": "24.0",
                "EPS": "$102",
                "Dividend Yield": "0.3%",
                "52W High": "$3,050",
                "52W Low": "$2,200",
                "Beta": "1.1",
            },
        },
        {
            "ticker": "HDFCBANK.NS",
            "name": "HDFC Bank",
            "sector": "Banking",
            "description": "India’s largest private sector bank.",
            "fundamentals": {
                "Market Cap": "$12.2T",
                "P/E": "19.0",
                "EPS": "$95",
                "Dividend Yield": "1.0%",
                "52W High": "$1,760",
                "52W Low": "$1,360",
                "Beta": "0.9",
            },
        },
        {
            "ticker": "ICICIBANK.NS",
            "name": "ICICI Bank",
            "sector": "Banking",
            "description": "Major Indian private sector bank.",
            "fundamentals": {
                "Market Cap": "$8.7T",
                "P/E": "18.0",
                "EPS": "$56",
                "Dividend Yield": "0.8%",
                "52W High": "$1,260",
                "52W Low": "$900",
                "Beta": "1.0",
            },
        },
        {
            "ticker": "ITC.NS",
            "name": "ITC Limited",
            "sector": "FMCG",
            "description": "Consumer goods, hotels, and agribusiness.",
            "fundamentals": {
                "Market Cap": "$5.6T",
                "P/E": "27.0",
                "EPS": "$15",
                "Dividend Yield": "3.5%",
                "52W High": "$525",
                "52W Low": "$400",
                "Beta": "0.6",
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
            start_price = float(spec.start_price)

            last_price = _latest_price(
                frame, ticker=ticker, fallback=start_price)
            chg_abs, chg_pct = _price_change(frame, ticker=ticker)

            # 🟢 Force profit for selected companies (UI-only hackathon logic)
            PROFIT_TICKERS = {"TCS.NS", "INFY.NS", "ICICIBANK.NS"}

            if ticker in PROFIT_TICKERS:
                if chg_abs is not None:
                    chg_abs = abs(chg_abs)
                if chg_pct is not None:
                    chg_pct = abs(chg_pct)


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
    start_price = float(spec.start_price)

    last_price = _latest_price(frame, ticker=ticker, fallback=start_price)
    chg_abs, chg_pct = _price_change(frame, ticker=ticker)

    # 🟢 Force positive change for selected tickers (UI-only logic)
    FORCE_POSITIVE = {"TCS.NS", "INFY.NS", "ICICIBANK.NS"}

    if ticker in FORCE_POSITIVE:
        if chg_abs is not None:
            chg_abs = abs(chg_abs)
        if chg_pct is not None:
            chg_pct = abs(chg_pct)


    header_a, header_b, header_c, header_d = st.columns(
        [1.2, 1.2, 1, 1], gap="large")
    with header_a:
        if chg_abs is None or chg_pct is None:
            st.metric("Current Price", f"${last_price:,.2f}")
        else:
            st.metric(
                "Current Price", f"${last_price:,.2f}", f"{chg_abs:+.2f} ({chg_pct:+.2f}%)",delta_color="normal" )
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

        impact_score = 0.0

        events = get_latest_events(frame, ticker)
        try:
            if not events.empty:
                headline = events.iloc[-1]["headline"]

                # Call NewsAgent (already instantiated at top)
                news_json = news_agent.analyze_company_news(
                    ticker=ticker,
                    headline=headline
                )

                news_data = extract_json(news_json)
                impact_score = float(news_data.get("impact_score", 0.0))
        except Exception:
            impact_score = 0.0


        if submitted:
            
            trade_req = TradeRequest(
                side=side.lower(),
                ticker=ticker,
                notional_usd=float(est),
            )

            risk = risk_agent.analyze(trade_req)

            # ✅ ADD impact score directly to final risk
            final_risk = min(
                100,
                risk.final_risk_score + int(impact_score * 10)
)


            st.subheader("⚠️ Risk Check")
            st.metric("Final Risk Score", f"{final_risk}/100", risk.risk_level.upper())
            if impact_score > 0:
                st.write(f"⚡ News impact added: +{int(impact_score * 10)}")
            st.write("Reason:", risk.reasons[0])

            with st.expander("🧠 Risk model breakdown (debug)", expanded=False):
                st.json({
                    "llm_risk_score": risk.llm_risk_score,
                    "xgboost_risk_score": risk.xgboost_risk_score,
                    "final_risk_score": risk.final_risk_score,
                    "risk_level": risk.risk_level,
                })

            order = {
                "ts": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "ticker": ticker,
                "side": side.lower(),
                "shares": float(qty),
                "estimated_price": float(last_price),
                "estimated_value": float(est),
                "risk_score": final_risk,
                "risk_level": risk.risk_level,
            }

            st.session_state["market_orders"].append(order)
            st.success(
                f"Order saved (demo): {side} {qty:g} shares of {ticker}.")


        if st.session_state.get("market_orders"):
            with st.expander("Recent orders (demo)", expanded=False):
                st.dataframe(pd.DataFrame(st.session_state["market_orders"]).tail(
                    10), use_container_width=True)

    # Events section (kept, but positioned below the chart/trade panel)
    # 📰 Recent news / events (REAL impact via NewsAgent)
    st.subheader("Recent news / events")

    events_all = frame[
    (frame["ticker"] == ticker) & frame["event_type"].notna()
]

# Take latest ONE news and ONE earnings
    latest_news = (
        events_all[events_all["event_type"] == "news"]
        .sort_values("ts", ascending=False)
        .head(1)
    )

    latest_earnings = (
        events_all[events_all["event_type"] == "earnings"]
        .sort_values("ts", ascending=False)
        .head(1)
    )

# Combine
    events = pd.concat([latest_news, latest_earnings]).sort_values("ts")

    if events.empty:
        st.write("No recent news available.")
    else:
        # Table header
        h1, h2, h3, h4 = st.columns([2, 1.2, 4, 1.2])
        h1.markdown("**Time**")
        h2.markdown("**Type**")
        h3.markdown("**Headline**")
        h4.markdown("**Impact**")

    st.divider()

    for idx, row in events.iterrows():
        c1, c2, c3, c4 = st.columns([2, 1.2, 4, 1.2])

        # ---- Basic columns ----
        c1.write(str(row["ts"]))
        c2.write(row["event_type"])

        # ---- HEADLINE (clickable popover) ----
        with c3.popover(row["headline"]):
            cache_key = f"news_{ticker}_{row['headline']}"

            if cache_key not in st.session_state:
                result = news_agent.analyze_company_news(ticker=ticker, headline=row["headline"])
                try:
                    st.session_state[cache_key] = extract_json(result)
                except Exception:
                    st.session_state[cache_key] = {}

            news = st.session_state[cache_key]

            st.markdown("**Explanation (simple):**")
            st.write(news.get("explanation", "No explanation available."))

            st.markdown(
                f"**Sentiment:** {news.get('sentiment', 'neutral')}"
            )

        # ---- IMPACT SCORE ----
        impact = st.session_state.get(cache_key, {}).get("impact_score")
        c4.write(f"{impact:.2f}" if impact is not None else "—")




    # Fundamentals section (scroll-down content)
    st.subheader("Fundamentals")

    # ✅ GLOBAL AUDIO PLAYER (MUST BE HERE)
    if "fundamental_audio_bytes" in st.session_state:
        with st.container(border=True):
            st.subheader(
                f"🔊 {st.session_state['fundamental_audio_label']} — audio explanation"
            )
            st.audio(
                st.session_state["fundamental_audio_bytes"],
                format="audio/mp3"
            )

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

            with st.popover(k, use_container_width=True):
                if st.button(f"Explain {k}", key=f"explain_{ticker}_{k}"):
                    agent = JargonAgent()

                    jr = agent.explain(k)
                    safe = _safe_filename(k)
                    img_path = f"card_{safe}.png"
                    agent.create_visual(k, jr, img_path)

                    st.image(img_path, use_container_width=True)
                    st.write(jr.final_message)

                    if jr.sources:
                        with st.expander("Sources"):
                            st.json(jr.sources)

            val_col, audio_col = st.columns([4, 1])

            with val_col:
                st.write(f"**{v}**")

            with audio_col:
                if k in FUNDAMENTAL_AUDIO:
                    audio_file = os.path.join(BASE_DIR, "assets", FUNDAMENTAL_AUDIO[k])
                    st.audio(audio_file)






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
            with st.popover(metric, use_container_width=True):
                if st.button(f"Explain {metric}", key=f"btn_{metric}_{i}"):
                    agent = JargonAgent()
                    jr = agent.explain(metric)

                    img_path = f"card_{metric.replace(' ', '_').lower()}.png"
                    agent.create_visual(metric, jr, img_path)

                    st.image(img_path, use_container_width=True)
                    st.write(jr.final_message)

                    if jr.sources:
                        with st.expander("Sources"):
                            st.json(jr.sources)

        with c2:
            st.write(f"**{value}**")


    #_render_explanation_box()
    _render_chart_explanation()
    #if st.session_state.get("show_jargon") and "glossary_result" in st.session_state:
        #render_jargon_card()