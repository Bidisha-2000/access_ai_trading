import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import joblib

def data_load_and_preprocess():
    # --------------------------------------------------------------
    # DOWNLOAD DATA
    # --------------------------------------------------------------
    tickers = [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "ASIANPAINT.NS",
        "TITAN.NS", "WIPRO.NS", "MARUTI.NS", "ULTRACEMCO.NS", "SUNPHARMA.NS"
    ]

    print("Downloading all tickers...")
    data = yf.download(
        tickers,
        start="2018-01-01",
        end="2025-11-30",
        group_by="ticker"
    )

    # --------------------------------------------------------------
    # FLATTEN MULTI-INDEX COLUMNS
    # --------------------------------------------------------------
    data.columns = [f"{col[0]}_{col[1]}" for col in data.columns]

    # --------------------------------------------------------------
    # BUILD LONG FORMAT DATAFRAME
    # --------------------------------------------------------------
    rows = []

    for ticker in tickers:

        temp = pd.DataFrame({
            "Date": data.index,
            "Ticker": ticker,
            "Open": data[f"{ticker}_Open"],
            "High": data[f"{ticker}_High"],
            "Low": data[f"{ticker}_Low"],
            "Close": data[f"{ticker}_Close"],
            "Volume": data[f"{ticker}_Volume"]
        })

        rows.append(temp)

    df = pd.concat(rows).reset_index(drop=True)

    print(df.head())
    print("Shape:", df.shape)

    # --------------------------------------------------------------
    # INDICATORS
    # --------------------------------------------------------------

    df["returns"] = df.groupby("Ticker")["Close"].pct_change()
    df["volatility"] = df.groupby("Ticker")["returns"].rolling(20).std().reset_index(0, drop=True)

    def rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        return 100 - (100 / (1 + rs))

    df["rsi"] = df.groupby("Ticker")["Close"].transform(rsi)

    def macd(series, fast=12, slow=26, signal=9):
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        return ema_fast - ema_slow

    df["macd"] = df.groupby("Ticker")["Close"].transform(macd)

    df["sma20"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(20).mean())
    df["sma50"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(50).mean())

    df["risk_label"] = pd.qcut(df["volatility"], 3, labels=["low", "medium", "high"])

    print(df.tail())

    return df

# -------------------------------
# Prepare data
# -------------------------------
def xgb_classifier():

    df = data_load_and_preprocess()

    features = [
        "returns", "volatility", "rsi", "macd", "sma20", "sma50", "Close"
    ]

    df_model = df.dropna(subset=features + ["risk_label"]).copy()

    X = df_model[features]
    y = df_model["risk_label"]

    # Encode labels (low=0, medium=1, high=2)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # -------------------------------
    # Train → Validation → Test split
    # -------------------------------

    # First split: Train (70%) vs Temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=0.30, random_state=42, shuffle=True
    )

    # Second split: Validation (15%) vs Test (15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, shuffle=True
    )

    print("Train size:", X_train.shape)
    print("Validation size:", X_val.shape)
    print("Test size:", X_test.shape)

    # -------------------------------
    # XGBoost model with validation
    # -------------------------------
    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss"
    )

    print("Training XGBoost model with validation...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],    # VALIDATION SET
        verbose=True
    )

    # -------------------------------
    # Final Evaluation on Test Set
    # -------------------------------
    y_pred = model.predict(X_test)
    
    print("\nFinal Test Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=encoder.classes_))

    # -------------------------------
    # Save the model + encoder
    # -------------------------------
    joblib.dump(model, "risk_xgboost_model.pkl")
    joblib.dump(encoder, "risk_label_encoder.pkl")

    print("\nModel saved as risk_xgboost_model.pkl")


def main():
    xgb_classifier()
    
if __name__ == "__main__":
    main()