# Positive factors (valid):

# ✔ MA bias aligns with trend often
# ✔ Hedge asymmetry gives slight edge
# ✔ TP and SL are close → many cycles
# ✔ BTC moves a lot → high volatility helps

# Invalid / inflated factors (bug/improvement needed):

# ⚠ Forward-scan until BOTH legs close → unrealistic fill sequencing
# ⚠ Time to close is ignored: SL/TP can hit many candles later
# ⚠ Many trades overlap in real world (not simulated)
# ⚠ No spread/slippage/fee
# ⚠ Winning trades naturally happen earlier in MA trend
import requests
import pandas as pd
import numpy as np
import time
import urllib3
import yfinance as yf

def get_price(symbol="BTC-USD", interval="1h", limit=1000):
    tf_map = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "1h": "60m",
        "4h": "1h",
        "1d": "1d"
    }

    interval_yf = tf_map.get(interval, "60m")

    df = yf.download(
        tickers=symbol,
        interval=interval_yf,
        period="60d",
        auto_adjust=True  # Suppress the FutureWarning
    )

    # Reset index to convert datetime index to a column
    df = df.reset_index()

    # Rename columns to match expected format
    df = df.rename(columns={
        "Datetime": "time",
        "Date": "time",  # yfinance uses "Date" for daily data
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close"
    })

    # Select columns and reset index to ensure clean integer index starting from 0
    result = df[["time", "open", "high", "low", "close"]].tail(limit).reset_index(drop=True)
    return result

# ===========================================================
# LOAD DATA FROM BINANCE
# ===========================================================
df = get_price(
    symbol="BTC-USD",
    interval="1h",
    limit=1000
)

print("Loaded candles:", len(df))


# ===========================================================
# INDICATORS (change or add freely)
# ===========================================================
df["fast_ma"] = df["close"].rolling(20).mean()
df["slow_ma"] = df["close"].rolling(50).mean()


# ===========================================================
# BIAS FUNCTION (EDIT TO TRY ANY INDICATOR)
# ===========================================================
def get_bias(df, i):
    fast = df["fast_ma"].iloc[i]
    slow = df["slow_ma"].iloc[i]
    
    if pd.isna(fast) or pd.isna(slow):
        return None

    # Bullish → fast MA > slow MA
    if fast > slow:
        return "bull"
    else:
        return "bear"


# ===========================================================
# BACKTEST PARAMETERS
# ===========================================================
POSITION_SIZE = 1

TP_BIG   = 12
SL_SMALL = 10

TP_SMALL = 10
SL_BIG   = 12

results = []


# ===========================================================
# BACKTEST LOOP
# ===========================================================
for i in range(len(df)):
    
    bias = get_bias(df, i)
    if bias is None:
        continue

    entry = float(df["close"].iloc[i])

    # TP/SL logic based on bias direction
    if bias == "bull":
        # Long gets bigger TP
        long_tp  = entry + TP_BIG
        long_sl  = entry - SL_SMALL
        short_tp = entry - TP_SMALL
        short_sl = entry + SL_BIG
    else:
        # Short gets bigger TP
        long_tp  = entry + TP_SMALL
        long_sl  = entry - SL_BIG
        short_tp = entry - TP_BIG
        short_sl = entry + SL_SMALL

    long_out = None
    short_out = None

    # Simulate forward candles until both sides close
    for j in range(i+1, len(df)):
        high = float(df["high"].iloc[j])
        low  = float(df["low"].iloc[j])

        # Long evaluation
        if long_out is None:
            if high >= long_tp:
                long_out = (long_tp - entry) * POSITION_SIZE
            elif low <= long_sl:
                long_out = (long_sl - entry) * POSITION_SIZE

        # Short evaluation
        if short_out is None:
            if low <= short_tp:
                short_out = (entry - short_tp) * POSITION_SIZE
            elif high >= short_sl:
                short_out = (entry - short_sl) * POSITION_SIZE

        # When both trades closed → record result
        if long_out is not None and short_out is not None:
            results.append(long_out + short_out)
            break


# ===========================================================
# RESULTS SUMMARY
# ===========================================================
if len(results) > 0:
    print("\n===== BACKTEST RESULT =====")
    print("Total Trades:", len(results))
    print(f"Net P/L: ${np.sum(results):.2f}")
    print(f"Average P/L: ${np.mean(results):.2f}")
    print(f"Win Rate: {(np.sum([1 for r in results if r > 0]) / len(results)) * 100:.2f}%")
else:
    print("\n===== NO TRADES EXECUTED =====")