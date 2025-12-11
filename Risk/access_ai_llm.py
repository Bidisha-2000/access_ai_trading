# ---------------------------------------------------------
# INSTALLS (uncomment during first run)
# ---------------------------------------------------------
# pip install langchain langchain-google-genai google-genai
# pip install joblib xgboost numpy pandas scikit-learn

# ---------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import joblib
import numpy as np
import json
import re
import os
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------
# LOAD XGBOOST MODEL + LABEL ENCODER
# ---------------------------------------------------------
xgb_model = joblib.load("risk_xgboost_model.pkl")
label_encoder = joblib.load("risk_label_encoder.pkl")

# ---------------------------------------------------------
# LLM MODEL (LangChain + Gemini)
# ---------------------------------------------------------
def get_llm(api_key):
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=api_key,
    )

# ---------------------------------------------------------
# LLM RISK PROMPT TEMPLATE
# ---------------------------------------------------------
risk_prompt = PromptTemplate(
    input_variables=["pe", "eps", "beta", "de", "vol", "user_action"],
    template="""
You are a financial safety assistant. 
Explain everything in very simple English using short sentences.
Avoid jargon. Avoid difficult words.

Your job:
1. Look at the stock fundamentals provided.
2. Think step-by-step.
3. Assess the risk level of buying this stock.
4. Give a confidence score between 0 and 1.
5. Provide a short, simple explanation suitable for a cognitive or learning-disabled user.
6. Do NOT recommend buy or sell.

Return output ONLY in this JSON format:

{{
 "llm_risk_score": <number between 0 and 1>,
 "explanation": "<short simple English>"
}}

Fundamentals:
P/E Ratio: {pe}
EPS: {eps}
Beta: {beta}
Debt/Equity: {de}
Volatility (1 year): {vol}
User Action: {user_action}
"""
)

# ---------------------------------------------------------
# CLEAN + PARSE JSON SAFELY
# ---------------------------------------------------------
def extract_json(text):
    """
    Removes backticks and extracts the pure JSON object.
    Works for Gemini output like:
    ```json
    { ... }
    ```
    """
    # Remove ```json and ``` 
    clean = re.sub(r"```json|```", "", text).strip()

    # Extract only the JSON object using regex
    json_match = re.search(r"\{[\s\S]*\}", clean)
    if json_match:
        return json.loads(json_match.group())
    
    # If still not found → fail gracefully
    raise ValueError("JSON not found in LLM response")

# ---------------------------------------------------------
# LLM RISK ANALYSIS USING LANGCHAIN
# ---------------------------------------------------------
def llm_risk_analysis(pe, eps, beta, de, vol, user_action, api_key):
    llm = get_llm(api_key)
    prompt = risk_prompt.format(
        pe=pe, eps=eps, beta=beta, de=de, vol=vol, user_action=user_action
    )

    response = llm.invoke(prompt).content
    print("\nLLM RAW OUTPUT:\n", response)

    try:
        parsed = extract_json(response)
        return float(parsed["llm_risk_score"]), parsed["explanation"]

    except Exception as e:
        print("\nJSON PARSE ERROR:", e)
        return 0.5, "I could not read the data clearly, so I gave a neutral score."

# ---------------------------------------------------------
# CLASSICAL (XGBOOST) RISK SCORE
# ---------------------------------------------------------
def classical_model_risk(feature_dict):
    X = np.array([[
        feature_dict["returns"],
        feature_dict["volatility"],
        feature_dict["rsi"],
        feature_dict["macd"],
        feature_dict["sma20"],
        feature_dict["sma50"],
        feature_dict["Close"]
    ]])

    proba = xgb_model.predict_proba(X)[0]
    class_indices = np.arange(len(proba))
    return float(np.sum(proba * class_indices) / 2.0)  # normalize 0–1


def llm_decide_final_risk(llm_score, classical_score, fundamentals, technicals, user_action, api_key):
    
    llm = get_llm(api_key)

    prompt = f"""
You are a financial safety assistant.

You will be given:
1. A risk score calculated by an LLM
2. A risk score calculated by a classical ML model
3. Some fundamental and technical indicators

Your job:
- Combine the two scores in a smart way
- You may weigh them however you think is appropriate
- Produce a final_risk_score between 0 and 1
- Label it as "low", "medium", or "high"
- Explain the result in simple English for a cognitive-disabled user

Output ONLY this JSON format:

{{
 "final_risk_score": <number between 0 and 1>,
 "risk_level": "<low/medium/high>",
 "alert_message": "<one short, simple sentence>"
}}

LLM Risk Score: {llm_score}
Classical Model Risk Score: {classical_score}

Fundamentals:
P/E: {fundamentals["pe"]}
EPS: {fundamentals["eps"]}
Beta: {fundamentals["beta"]}
Debt/Equity: {fundamentals["de"]}
Volatility: {fundamentals["vol"]}

Technicals:
Returns: {technicals["returns"]}
Volatility: {technicals["volatility"]}
RSI: {technicals["rsi"]}
MACD: {technicals["macd"]}
SMA20: {technicals["sma20"]}
SMA50: {technicals["sma50"]}
Close: {technicals["Close"]}

User Action: "{user_action}"
"""

    response = llm.invoke(prompt).content

    # Clean JSON if needed
    clean = response.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(clean)

    return (
        parsed["final_risk_score"],
        parsed["risk_level"],
        parsed["alert_message"]
    )


# ---------------------------------------------------------
# HYBRID RISK AGENT (LLM + Classical)
# ---------------------------------------------------------
def hybrid_risk_agent(fundamentals, technicals, user_action, api_key):

    llm_score, llm_explanation = llm_risk_analysis(
        fundamentals["pe"], fundamentals["eps"], fundamentals["beta"],
        fundamentals["de"], fundamentals["vol"], user_action, api_key
    )

    classical_score = classical_model_risk(technicals)

    final_risk, risk_level, alert = llm_decide_final_risk(
    llm_score, classical_score, fundamentals, technicals, user_action, api_key
)

    return {
        "llm_score": llm_score,
        "classical_score": classical_score,
        "final_risk": final_risk,
        "risk_level": risk_level,
        "alert": alert
    }



# ---------------------------------------------------------
# EXAMPLE RUN
# ---------------------------------------------------------
if __name__ == "__main__":

    API_KEY = os.getenv("MY_TOKEN")

    test_cases = {
        "LOW RISK": {
            "fundamentals": {
                "pe": 18.0,
                "eps": 25.4,
                "beta": 0.85,
                "de": 0.20,
                "vol": 0.10
            },
            "technicals": {
                "returns": 0.012,
                "volatility": 0.08,
                "rsi": 45,
                "macd": 1.5,
                "sma20": 1020,
                "sma50": 1015,
                "Close": 1022
            },
            "user_action": "I want to buy this stock"
        },

        "MEDIUM RISK": {
            "fundamentals": {
                "pe": 28.0,
                "eps": 10.2,
                "beta": 1.1,
                "de": 0.55,
                "vol": 0.20
            },
            "technicals": {
                "returns": -0.003,
                "volatility": 0.18,
                "rsi": 58,
                "macd": 0.8,
                "sma20": 980,
                "sma50": 1000,
                "Close": 985
            },
            "user_action": "Should I buy this?"
        },

        "HIGH RISK": {
            "fundamentals": {
                "pe": 65.0,
                "eps": 3.5,
                "beta": 1.8,
                "de": 1.2,
                "vol": 0.40
            },
            "technicals": {
                "returns": -0.025,
                "volatility": 0.35,
                "rsi": 78,
                "macd": -1.2,
                "sma20": 1120,
                "sma50": 1150,
                "Close": 1105
            },
            "user_action": "I want to buy this stock"
        }
    }

    # -------- Run all test cases --------
    for label, data in test_cases.items():
        print(f"\n\n===================== {label} =====================")

        result = hybrid_risk_agent(
            data["fundamentals"],
            data["technicals"],
            data["user_action"],
            API_KEY
        )

        print("\nOUTPUT:\n", result)
