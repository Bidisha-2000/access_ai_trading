from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import streamlit as st

from omago_ai.agents.jargon_agent import JargonAgent
from omago_ai.orchestrator.router import LeadRouter
from omago_ai.orchestrator.state import SessionState
from omago_ai.market import (
    MarketSimulator,
    SimulatorConfig,
    build_market_snapshot,
    default_universe,
)
from omago_ai.market.utils import ticks_to_frame
from omago_ai.llm import get_openai_client_from_env


# -------------------------------------------------
# Streamlit config
# -------------------------------------------------
st.set_page_config(page_title="OmagoAI Prototype", layout="wide")

st.title("OmagoAI – Accessible Trading Companion (Prototype)")
st.caption("Education-only prototype. Not financial advice.")

# -------------------------------------------------
# Session defaults
# -------------------------------------------------
if "show_jargon_modal" not in st.session_state:
    st.session_state["show_jargon_modal"] = False

# -------------------------------------------------
# TRUE MODAL (Streamlit 1.52 compatible)
# -------------------------------------------------
@st.dialog("📘 Learn more")
def show_jargon_modal(data):
    st.image(data["image"], width=500)

    st.markdown("### 🎧 Audio explanation")
    st.audio(data["audio"])

    with st.expander("📝 Read narration text"):
        st.write(data["narration"])

    if st.button("Close"):
        st.session_state["show_jargon_modal"] = False
        st.rerun()


# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.subheader("User Preferences")
    reading_level = st.selectbox(
        "Reading level",
        options=["simple", "standard"],
        index=0,
    )
    st.session_state["reading_level"] = reading_level

# -------------------------------------------------
# Tabs
# -------------------------------------------------
tab_assistant, tab_market = st.tabs(["Assistant", "Market Simulation"])

# =================================================
# ASSISTANT TAB
# =================================================
with tab_assistant:
    st.divider()
    col1, col2 = st.columns([2, 1], gap="large")

    # ---------------------------------------------
    # LEFT: Main assistant
    # ---------------------------------------------
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
                    reading_level=st.session_state.get("reading_level", "simple"),
                    market_snapshot=build_market_snapshot(market_ticks)
                    if market_ticks
                    else None,
                )

                llm_client = None
                use_llm = bool(send_llm)

                if use_llm:
                    try:
                        llm_client = get_openai_client_from_env()
                    except Exception as e:
                        st.error(f"LLM mode not configured.\n\n{e}")
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

            # JSON output (for Risk Agent / Debug / Hackathon judges)
            with st.expander("🔍 View raw JSON output"):
                st.json(st.session_state["last_result"].model_dump())

            if getattr(st.session_state["last_result"], "sources", None):
                with st.expander("Sources"):
                    st.json(st.session_state["last_result"].sources)

    # ---------------------------------------------
    # RIGHT: GLOSSARY (WITH TRUE POPOUT)
    # ---------------------------------------------
    with col2:
        st.subheader("Glossary search")

        q = st.text_input(
            "Type one trading term",
            placeholder="RSI, EMA, VWAP, support, resistance…",
        )

        if st.button("Search", key="glossary_search"):
            if not q.strip():
                st.warning("Please type a trading term.")
            else:
                agent = JargonAgent()
                jr = agent.explain(q)

                st.session_state["glossary_term"] = q
                st.session_state["glossary_result"] = jr

        if "glossary_result" in st.session_state:
            jr = st.session_state["glossary_result"]
            term = st.session_state["glossary_term"]
            explanation = jr.final_message

            st.markdown("### Simple explanation")

            st.markdown(
                f"""
                <div style="
                    padding:16px;
                    border-radius:12px;
                    background-color:#f6f8fa;
                    border:1px solid #e1e4e8;
                    font-size:16px;
                    line-height:1.6;
                ">
                    <strong>{term}</strong><br><br>
                    {explanation}
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("🔊 Want to know more"):
                agent = JargonAgent()

                narration = agent.narrate(term)
                image_path = agent.create_visual(term, explanation)
                audio_path = agent.text_to_speech(narration)

                st.session_state["jargon_modal_data"] = {
                    "term": term,
                    "image": image_path,
                    "audio": audio_path,
                    "narration": narration,
                }
                st.session_state["show_jargon_modal"] = True

        # Trigger modal
        if st.session_state.get("show_jargon_modal"):
            show_jargon_modal(st.session_state["jargon_modal_data"])

# =================================================
# MARKET TAB
# =================================================
with tab_market:
    st.divider()
    st.subheader("Live-ish market feed (simulated)")
    st.caption("This sim is for demo only.")

    if "market_sim" not in st.session_state:
        st.session_state["market_sim"] = MarketSimulator(
            universe=default_universe(),
            config=SimulatorConfig(
                seed=7,
                tick_seconds=1.0,
                event_probability_per_tick=0.02,
            ),
            start_ts=datetime.now(timezone.utc).replace(microsecond=0),
        )
        st.session_state["market_ticks"] = []

    sim: MarketSimulator = st.session_state["market_sim"]

    col_a, col_b, col_c = st.columns([1, 1, 2], gap="large")

    with col_a:
        if st.button("Tick once"):
            st.session_state["market_ticks"].extend(sim.step())

        if st.button("Run 60 ticks"):
            st.session_state["market_ticks"].extend(sim.run(60))

        if st.button("Reset simulator"):
            st.session_state.pop("market_sim", None)
            st.session_state.pop("market_ticks", None)
            st.rerun()

    ticks = st.session_state.get("market_ticks", [])
    frame = (
        ticks_to_frame(ticks)
        if ticks
        else pd.DataFrame(columns=["ts", "ticker", "price", "volume"])
    )

    with col_b:
        ticker = st.selectbox(
            "Ticker",
            options=[t.ticker for t in default_universe()],
            index=0,
        )

        if not frame.empty:
            latest = frame[frame["ticker"] == ticker].tail(1)
            if not latest.empty:
                st.metric("Last price", f"{float(latest.iloc[0]['price']):.2f}")

    with col_c:
        if not frame.empty:
            sub = frame[frame["ticker"] == ticker].tail(240)
            if not sub.empty:
                fig = px.line(
                    sub,
                    x="ts",
                    y="price",
                    title=f"{ticker} price (simulated)",
                )
                fig.update_layout(
                    margin=dict(l=10, r=10, t=40, b=10),
                    height=340,
                )
                st.plotly_chart(fig, use_container_width=True)
