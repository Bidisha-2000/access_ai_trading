from __future__ import annotations

# ---------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------
import os
import json
import re
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from omago_ai.orchestrator.parsing import TradeRequest

load_dotenv()

# ---------------------------------------------------------
# DATA MODEL (RETURNED TO UI / ORCHESTRATOR)
# ---------------------------------------------------------
class RiskResult(BaseModel):
    llm_risk_score: float = Field(ge=0.0, le=1.0)
    xgboost_risk_score: float = Field(ge=0.0, le=1.0)

    final_risk_score: int = Field(ge=0, le=100)
    risk_level: str

    reasons: list[str]
    safer_actions: list[str]

    # backward compatibility
    @property
    def risk_score(self) -> int:
        return self.final_risk_score


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "risk_xgboost_model.pkl"
ENCODER_PATH = BASE_DIR / "risk_label_encoder.pkl"
CSV_PATH = BASE_DIR / "nse_market_features.csv"

# ---------------------------------------------------------
# LOAD MODELS + CSV (ONCE)
# ---------------------------------------------------------
xgb_model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)
MARKET_DF = pd.read_csv(CSV_PATH)

# ---------------------------------------------------------
# HELPER: GET LATEST MARKET ROW
# ---------------------------------------------------------
def get_latest_row(ticker: str):
    rows = MARKET_DF[MARKET_DF["Ticker"] == ticker]
    if rows.empty:
        return None
    return rows.sort_values("Date").iloc[-1]


# ---------------------------------------------------------
# LLM CLIENT
# ---------------------------------------------------------
def get_llm(api_key: str):
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
        google_api_key=api_key,
    )


# ---------------------------------------------------------
# LLM FUNDAMENTAL PROMPT
# ---------------------------------------------------------
risk_prompt = PromptTemplate(
    input_variables=["pe", "eps", "beta", "de", "vol", "user_action"],
    template="""
You are a financial safety assistant.

Explain in very simple English.
Use short sentences. Avoid jargon.
IMPORTANT DEFINITIONS:
- Risk score means: probability of downside or loss
- It is NOT a qualitative opinion
- It must be comparable to a statistical ML model output

Your job:
- Look at the stock data
- Estimate the likelihood of downside based ONLY on the data
- Be conservative in numbers, not language
- Typical values should be between 0.05 and 0.40 for stable large-cap stocks
- Do NOT exaggerate risk
- Do NOT recommend buy or sell
- Ignore the investment amount completely
- Be deterministic. Do not infer anything not provided

Return ONLY JSON:

{{
  "llm_risk_score": <number between 0 and 1>,
  "explanation": "<simple explanation>"
}}

Data:
P/E: {pe}
EPS: {eps}
Beta: {beta}
Debt/Equity: {de}
Volatility: {vol}
User Action: {user_action}
"""
)

# ---------------------------------------------------------
# JSON EXTRACTION
# ---------------------------------------------------------
def extract_json(text: str) -> dict:
    clean = re.sub(r"```json|```", "", text).strip()
    match = re.search(r"\{[\s\S]*\}", clean)
    if not match:
        raise ValueError("JSON not found")
    return json.loads(match.group())


# ---------------------------------------------------------
# LLM FUNDAMENTAL RISK
# ---------------------------------------------------------
def llm_risk_analysis(pe, eps, beta, de, vol, user_action, api_key):
    llm = get_llm(api_key)

    prompt = risk_prompt.format(
        pe=pe,
        eps=eps,
        beta=beta,
        de=de,
        vol=vol,
        user_action=user_action,
    )

    response = llm.invoke(prompt).content
    try:
        parsed = extract_json(response)
        return float(parsed["llm_risk_score"]), parsed["explanation"]
    except Exception:
        return 0.5, "Risk is unclear, so I assumed a medium level."


# ---------------------------------------------------------
# CLASSICAL XGBOOST RISK
# ---------------------------------------------------------
def classical_model_risk(technicals: dict) -> float:
    X = np.array([[
        technicals["returns"],
        technicals["volatility"],
        technicals["rsi"],
        technicals["macd"],
        technicals["sma20"],
        technicals["sma50"],
        technicals["Close"],
    ]])

    proba = xgb_model.predict_proba(X)[0]
    idx = np.arange(len(proba))

    # expected risk ∈ [0,1]
    return float(np.sum(proba * idx) / 2.0)


# ---------------------------------------------------------
# FINAL LLM FUSION (STRICT AVERAGE)
# ---------------------------------------------------------
def llm_decide_final_risk(
    llm_score,
    classical_score,
    user_action,
    api_key,
):
    llm = get_llm(api_key)

    prompt = f"""
You are a financial safety assistant.

You will be given:
1. A risk score calculated by an LLM
2. A risk score calculated by a classical ML model
3. Some fundamental and technical indicators

FOLLOW THESE STEPS EXACTLY:
1. Compute:
   final_risk_score = (LLM risk score + ML risk score) / 2
2. Do NOT modify the numbers.
3. Do NOT add extra caution.
4. Use these fixed thresholds:
- low: final_risk_score < 0.33
- medium: 0.33 ≤ final_risk_score < 0.66
- high: final_risk_score ≥ 0.66


Then:
- Assign risk_level based on the final score.
- Write ONE short sentence for a cognitive user.

Return ONLY JSON:

{{
  "final_risk_score": <number between 0 and 1>,
  "risk_level": "<low/medium/high>",
  "alert_message": "<one short sentence>"
}}

LLM risk score = {llm_score}
Classical ML risk score = {classical_score}
User action = "{user_action}"
"""

    response = llm.invoke(prompt).content
    parsed = extract_json(response)

    return (
        float(parsed["final_risk_score"]),
        parsed["risk_level"],
        parsed["alert_message"],
    )


# ---------------------------------------------------------
# MAIN RISK AGENT
# ---------------------------------------------------------
class RiskAgent:
    """
    Hybrid Risk Agent:
    - CSV technicals
    - XGBoost probability
    - LLM explanation + fusion
    """

    def analyze(self, trade: TradeRequest, *, market=None) -> RiskResult:
        api_key = os.getenv("MY_TOKEN")

        row = get_latest_row(trade.ticker)

        if row is not None:
            technicals = {
                "returns": float(row["returns"]),
                "volatility": float(row["volatility"]),
                "rsi": 50.0,
                "macd": 0.0,
                "sma20": float(row["sma20"]),
                "sma50": float(row["sma50"]),
                "Close": float(row["Close"]),
            }

            fundamentals = {
                "pe": 28.0,
                "eps": 80.0,
                "beta": 0.9,
                "de": 0.3,
                "vol": technicals["volatility"],
            }
        else:
            technicals = {
                "returns": 0.0,
                "volatility": 0.25,
                "rsi": 50.0,
                "macd": 0.0,
                "sma20": 0.0,
                "sma50": 0.0,
                "Close": 0.0,
            }

            fundamentals = {
                "pe": 30.0,
                "eps": 50.0,
                "beta": 1.0,
                "de": 0.5,
                "vol": 0.25,
            }

        llm_score, llm_reason = llm_risk_analysis(
            fundamentals["pe"],
            fundamentals["eps"],
            fundamentals["beta"],
            fundamentals["de"],
            fundamentals["vol"],
            trade.side,
            api_key,
        )

        xgb_score = classical_model_risk(technicals)

        final_risk, level, alert = llm_decide_final_risk(
            llm_score,
            xgb_score,
            trade.side,
            api_key,
        )

        return RiskResult(
            llm_risk_score=round(llm_score, 3),
            xgboost_risk_score=xgb_score,
            final_risk_score=int(final_risk * 100),
            risk_level=level,
            reasons=[llm_reason],
            safer_actions=[alert],
        )


# ---------------------------------------------------------
# LOCAL TEST
# ---------------------------------------------------------
if __name__ == "__main__":
    trade = TradeRequest(
        side="buy",
        ticker="TCS.NS",
        notional_usd=5000,
    )

    agent = RiskAgent()
    result = agent.analyze(trade)

    print("\n=== RISK AGENT OUTPUT ===")
    print(result.model_dump())
