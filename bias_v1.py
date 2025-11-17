import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# -------------------------
# USER PARAMETERS
# -------------------------
SYMBOL = "BTC-USD"
INTERVAL = "1d"
LIMIT = 5760

INITIAL_CAPITAL = 20.0
LEVERAGE = 10.0
ALLOW_OVERLAP = False

TAKER_FEE_RATE = 0.0004
SLIPPAGE_PCT = 0.0002

RR_FOLLOW = 2.0

# -------------------------
# DATA DOWNLOAD
# -------------------------
def get_price(symbol="BTC-USD", interval="15m", limit=5760):
    tf_map = {"1m":"1m","5m":"5m","15m":"15m","1h":"1h","4h":"4h","1d":"1d"}
    interval_yf = tf_map.get(interval, "15m")
    df = yf.download(
        tickers=symbol,
        interval=interval_yf,
        period="60d",
        auto_adjust=False,
        progress=False
    )
    df = df.reset_index()
    df = df.rename(columns={"Datetime":"time","Date":"time","Open":"open","High":"high",
                            "Low":"low","Close":"close","Volume":"volume"})
    
    if "volume" not in df.columns:
        df["volume"] = 1

    df = df[["time","open","high","low","close","volume"]].tail(limit).reset_index(drop=True)
    df['time'] = pd.to_datetime(df['time'])
    return df

df = get_price(SYMBOL, INTERVAL, LIMIT)
if df.empty:
    raise SystemExit("No data downloaded. Check connectivity and ticker.")

# -------------------------
# INDICATORS & BIAS FUNCTION
# -------------------------
def compute_indicators(df):
    df["fast_ma"] = df["close"].rolling(20).mean()
    df["slow_ma"] = df["close"].rolling(50).mean()
    
    df["delta_close"] = df["close"].diff()
    df["obv"] = 0
    for i in range(1, len(df)):
        if df["delta_close"].iloc[i] > 0:
            df["obv"].iloc[i] = df["obv"].iloc[i-1] + df["volume"].iloc[i]
        elif df["delta_close"].iloc[i] < 0:
            df["obv"].iloc[i] = df["obv"].iloc[i-1] - df["volume"].iloc[i]
        else:
            df["obv"].iloc[i] = df["obv"].iloc[i-1]
    
    df["tr"] = df["high"] - df["low"]
    df["atr"] = df["tr"].rolling(14).mean()
    
    df["bull_bar"] = df["close"] > df["open"]
    df["bear_bar"] = df["close"] < df["open"]
    
    return df

def get_bias(df, i):
    score = 0
    
    if pd.isna(df["fast_ma"].iloc[i]) or pd.isna(df["slow_ma"].iloc[i]) or pd.isna(df["atr"].iloc[i]):
        return None

    if df["fast_ma"].iloc[i] > df["slow_ma"].iloc[i]:
        score += 1
    elif df["fast_ma"].iloc[i] < df["slow_ma"].iloc[i]:
        score -= 1
    
    if i > 1 and df["obv"].iloc[i] > df["obv"].iloc[i-1]:
        score += 1
    elif i > 1 and df["obv"].iloc[i] < df["obv"].iloc[i-1]:
        score -= 1
    
    if df["bull_bar"].iloc[i]:
        score += 1
    elif df["bear_bar"].iloc[i]:
        score -= 1
    
    if score >= 2:
        return "bull"
    elif score <= -2:
        return "bear"
    else:
        return None

def calculate_trade_levels(entry_price, last_high, last_low, bias, rr_follow=2.0):
    if bias == "bull":
        stop = last_low
        risk = entry_price - stop
        tp = entry_price + risk * rr_follow
    else:
        stop = last_high
        risk = stop - entry_price
        tp = entry_price - risk * rr_follow
    return {"tp": tp, "sl": stop}

df = compute_indicators(df)

# -------------------------
# FIXED BAR HIT HELPER
# -------------------------
def check_stops_in_bar(open_p, high_p, low_p, close_p, tp_level, sl_level, bias):
    """
    Check if TP or SL was hit in this bar.
    Returns: (hit_tp, hit_sl)
    
    For BULL trades:
    - TP is above entry (hit if high >= tp)
    - SL is below entry (hit if low <= sl)
    
    For BEAR trades:
    - TP is below entry (hit if low <= tp)
    - SL is above entry (hit if high >= sl)
    """
    hit_tp = False
    hit_sl = False
    
    if bias == "bull":
        # For long: check if we hit TP (above) or SL (below)
        if high_p >= tp_level:
            hit_tp = True
        if low_p <= sl_level:
            hit_sl = True
    else:  # bear
        # For short: check if we hit TP (below) or SL (above)
        if low_p <= tp_level:
            hit_tp = True
        if high_p >= sl_level:
            hit_sl = True
    
    # Determine which was hit first (assumes both can't be hit in same bar for proper R:R)
    # In reality, with proper risk management, both shouldn't trigger in one bar
    # But if they do, we prioritize SL (more conservative)
    if hit_sl and hit_tp:
        # If both hit, check which direction the bar moved first
        # Conservative: assume SL hit first
        return False, True
    
    return hit_tp, hit_sl

# -------------------------
# BACKTEST WITH FIXED LOGIC
# -------------------------
capital = INITIAL_CAPITAL
equity_curve = [capital]
results = []
legs_stats = {"win":0, "loss":0}
trade_num = 0

i = 1
pbar = tqdm(total=len(df), desc="Backtesting", unit="bar")
while i < len(df)-1:
    pbar.update(1)
    bias = get_bias(df, i)
    if bias is None:
        i += 1
        continue

    entry_price = float(df["open"].iloc[i+1])
    last_high = float(df["high"].iloc[i])
    last_low = float(df["low"].iloc[i])

    levels = calculate_trade_levels(entry_price, last_high, last_low, bias, RR_FOLLOW)

    size = (capital * LEVERAGE) / entry_price
    if size <= 0:
        i += 1
        continue

    entry_notional = entry_price * size
    fee_entry = entry_notional * TAKER_FEE_RATE
    capital -= fee_entry

    trade_open = True
    close_price = None
    exit_type = None

    j = i + 2
    while j < len(df) and trade_open:
        o = float(df["open"].iloc[j])
        h = float(df["high"].iloc[j])
        l = float(df["low"].iloc[j])
        c = float(df["close"].iloc[j])

        hit_tp, hit_sl = check_stops_in_bar(o, h, l, c, levels["tp"], levels["sl"], bias)

        # DEBUG: For first few trades, show what's happening
        if trade_num < 3 and j < i + 10:
            print(f"  Bar {j} ({df['time'].iloc[j]}): L={l:.2f}, H={h:.2f} | SL={levels['sl']:.2f}, TP={levels['tp']:.2f} | Hit TP: {hit_tp}, Hit SL: {hit_sl}")

        if hit_tp:
            close_price = levels["tp"] * ((1 + SLIPPAGE_PCT) if bias=="bull" else (1 - SLIPPAGE_PCT))
            exit_type = "tp"
            trade_open = False
            if trade_num < 3:
                print(f"  >>> TP HIT at {close_price:.2f}")
            break
        elif hit_sl:
            close_price = levels["sl"] * ((1 - SLIPPAGE_PCT) if bias=="bull" else (1 + SLIPPAGE_PCT))
            exit_type = "sl"
            trade_open = False
            if trade_num < 3:
                print(f"  >>> SL HIT at {close_price:.2f}")
            break
        j += 1

    if trade_open:
        final_price = float(df["close"].iloc[-1])
        close_price = final_price * ((1 + SLIPPAGE_PCT) if bias=="bull" else (1 - SLIPPAGE_PCT))
        exit_type = "mtm"
        if trade_num < 5:
            print(f"Trade still open at end - MTM at {close_price:.2f}")

    pl = (close_price - entry_price) * size if bias=="bull" else (entry_price - close_price) * size
    exit_fee = abs(close_price*size)*TAKER_FEE_RATE
    capital += pl
    capital -= exit_fee

    if pl > 0:
        legs_stats["win"] +=1
    else:
        legs_stats["loss"] +=1

    trade_num +=1
    results.append({
        "trade": trade_num,
        "entry_index": i+1,
        "entry_time": df["time"].iloc[i+1],
        "bias": bias,
        "size": size,
        "entry": entry_price,
        "stop_loss": levels['sl'],
        "take_profit": levels['tp'],
        "close_price": close_price,
        "exit_type": exit_type,
        "pl": pl,
        "fee_entry": fee_entry,
        "fee_exit": exit_fee,
        "net": pl - exit_fee
    })

    equity_curve.append(capital)
    i = j if not ALLOW_OVERLAP else i+1

pbar.close()

# -------------------------
# REPORT
# -------------------------
res_df = pd.DataFrame(results)
net_pls = res_df["net"].values if not res_df.empty else np.array([])

total_net = capital - INITIAL_CAPITAL
avg_pl = np.mean(net_pls) if len(net_pls)>0 else 0.0
median_pl = np.median(net_pls) if len(net_pls)>0 else 0.0
win_rate = np.mean([1 if x>0 else 0 for x in net_pls])*100 if len(net_pls)>0 else 0.0

equity = np.array(equity_curve)
running_max = np.maximum.accumulate(equity)
drawdowns = (running_max - equity)/running_max
max_dd = np.max(drawdowns) if len(drawdowns)>0 else 0.0

print("\n===== FIXED BACKTEST RESULTS =====")
print(f"Trades executed: {len(res_df)}")
print(f"Starting capital: ${INITIAL_CAPITAL:.2f}")
print(f"Ending capital:  ${capital:.2f}")
print(f"Total Net P/L:   ${total_net:.2f}")
print(f"Average trade P/L: ${avg_pl:.2f}")
print(f"Median trade P/L:  ${median_pl:.2f}")
print(f"Win rate: {win_rate:.2f}%")
print(f"Legs: wins={legs_stats['win']}, losses={legs_stats['loss']}")
print(f"Max drawdown: {max_dd*100:.2f}%")

plt.figure(figsize=(10,4))
plt.plot(equity, marker='o', linewidth=1, markersize=2)
plt.title('Equity Curve (Fixed Stop Loss Logic)')
plt.xlabel('Trade number')
plt.ylabel('Capital (USD)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

res_df.to_csv("bias_only_results_FIXED.csv", index=False)
print("\nSaved detailed trades to bias_only_results_FIXED.csv")