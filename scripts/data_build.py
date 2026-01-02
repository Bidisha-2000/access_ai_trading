import yfinance as yf
import pandas as pd

def data_load_and_preprocess():
    tickers = [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "ASIANPAINT.NS",
        "TITAN.NS", "WIPRO.NS", "MARUTI.NS", "ULTRACEMCO.NS", "SUNPHARMA.NS"
    ]

    data = yf.download(
        tickers,
        start="2024-01-01",
        end="2025-01-01",
        group_by="ticker"
    )

    data.columns = [f"{c[0]}_{c[1]}" for c in data.columns]

    rows = []
    for t in tickers:
        df = pd.DataFrame({
            "Date": data.index,
            "Ticker": t,
            "Close": data[f"{t}_Close"],
            "Volume": data[f"{t}_Volume"],
        })
        rows.append(df)

    df = pd.concat(rows).reset_index(drop=True)

    # indicators
    df["returns"] = df.groupby("Ticker")["Close"].pct_change()
    df["volatility"] = df.groupby("Ticker")["returns"].rolling(20).std().reset_index(0, drop=True)
    df["sma20"] = df.groupby("Ticker")["Close"].rolling(20).mean().reset_index(0, drop=True)
    df["sma50"] = df.groupby("Ticker")["Close"].rolling(50).mean().reset_index(0, drop=True)

    df.dropna(inplace=True)
    return df


if __name__ == "__main__":
    df = data_load_and_preprocess()
    df.to_csv("src/omago_ai/agents/nse_market_features.csv", index=False)
    print("CSV saved")
