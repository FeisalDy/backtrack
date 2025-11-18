# hedged_binance_futures_backtest_rr.py
# Realistic OCO hedged-pair backtester for Binance Futures VIP0 assumptions with custom risk-reward
# Requirements: yfinance, pandas, numpy, matplotlib
# Run: python hedged_binance_futures_backtest_rr.py

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# USER PARAMETERS
# -------------------------
SYMBOL = "BTC-USD"
INTERVAL = "15m"
LIMIT = 5760

# Starting capital & sizing
INITIAL_CAPITAL = 20.0
LEVERAGE = 10.0
ALLOW_OVERLAP = False

# Market realism settings
TAKER_FEE_RATE = 0.0004
SLIPPAGE_PCT = 0.0002

# Risk-reward settings
RR_FOLLOW = 2.0
RR_OPP = 0.6

# -------------------------
# DATA DOWNLOAD
# -------------------------
def get_price(symbol="BTC-USD", interval="15m", limit=5760):
    tf_map = {"1m":"1m","5m":"5m","15m":"15m","1h":"60m","4h":"240m","1d":"1d"}
    interval_yf = tf_map.get(interval, "15m")
    df = yf.download(tickers=symbol, interval=interval_yf, period="60d", auto_adjust=True, progress=False)
    df = df.reset_index()
    df = df.rename(columns={"Datetime":"time","Date":"time","Open":"open","High":"high","Low":"low","Close":"close"})
    df = df[["time","open","high","low","close"]].tail(limit).reset_index(drop=True)
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
    return df

def get_bias(df, i):
    """Return 'bull', 'bear' or None"""
    fast = df["fast_ma"].iloc[i]
    slow = df["slow_ma"].iloc[i]
    if pd.isna(fast) or pd.isna(slow):
        return None
    return "bull" if fast > slow else "bear"

def calculate_trade_levels(entry_price, last_high, last_low, bias, rr_follow=2.0, rr_opp=0.6):
    if bias == "bull":
        stop_follow = last_low
        risk_follow = entry_price - stop_follow
        tp_follow = entry_price + risk_follow * rr_follow

        risk_opp = stop_follow - entry_price
        tp_opp = entry_price - abs(risk_opp) * rr_opp
        stop_opp = entry_price + abs(risk_opp)
    else:
        stop_follow = last_high
        risk_follow = stop_follow - entry_price
        tp_follow = entry_price - risk_follow * rr_follow

        risk_opp = entry_price - stop_follow
        tp_opp = entry_price + abs(risk_opp) * rr_opp
        stop_opp = entry_price - abs(risk_opp)

    return {
        "long": {"tp": tp_follow if bias=="bull" else tp_opp,
                 "sl": stop_follow if bias=="bull" else stop_opp},
        "short":{"tp": tp_opp if bias=="bull" else tp_follow,
                 "sl": stop_opp if bias=="bull" else stop_follow}
    }

df = compute_indicators(df)

# -------------------------
# BAR HIT HELPER
# -------------------------
def check_hit_in_bar(open_p, high_p, low_p, close_p, target):
    if open_p <= target <= high_p:
        return True, target
    if low_p <= target <= high_p:
        return True, target
    if min(low_p, close_p) <= target <= max(low_p, close_p):
        return True, target
    return False, None

# -------------------------
# BACKTEST
# -------------------------
capital = INITIAL_CAPITAL
equity_curve = [capital]
results = []
legs_stats = {"long_win":0, "long_loss":0, "short_win":0, "short_loss":0}
trade_num = 0

i = 1  # start from 1 to have last bar for SL
while i < len(df):
    bias = get_bias(df, i)
    if bias is None:
        i += 1
        continue

    entry_price = float(df["close"].iloc[i])
    last_high = float(df["high"].iloc[i-1])
    last_low = float(df["low"].iloc[i-1])

    trade_levels = calculate_trade_levels(entry_price, last_high, last_low, bias, RR_FOLLOW, RR_OPP)

    size = (capital * LEVERAGE) / entry_price
    if size <= 0:
        i += 1
        continue

    entry_notional = entry_price * size
    fee_entry_per_leg = entry_notional * TAKER_FEE_RATE
    total_entry_fees = 2 * fee_entry_per_leg
    capital -= total_entry_fees

    long_open = True
    short_open = True
    long_close_price = None
    short_close_price = None
    long_exit_type = None
    short_exit_type = None

    j = i + 1
    while j < len(df) and (long_open or short_open):
        o = float(df["open"].iloc[j])
        h = float(df["high"].iloc[j])
        l = float(df["low"].iloc[j])
        c = float(df["close"].iloc[j])

        # LONG check
        if long_open:
            hit_tp, _ = check_hit_in_bar(o,h,l,c,trade_levels["long"]["tp"])
            hit_sl, _ = check_hit_in_bar(o,h,l,c,trade_levels["long"]["sl"])
            if hit_tp:
                long_close_price = trade_levels["long"]["tp"] * (1 + SLIPPAGE_PCT)
                long_exit_type = "tp"
                long_open = False

                short_close_price = long_close_price
                short_exit_type = "oco_due_to_long"
                short_open = False
                break
            elif hit_sl:
                long_close_price = trade_levels["long"]["sl"] * (1 - SLIPPAGE_PCT)
                long_exit_type = "sl"
                long_open = False

                short_close_price = long_close_price
                short_exit_type = "oco_due_to_long"
                short_open = False
                break

        # SHORT check
        if short_open:
            hit_tp, _ = check_hit_in_bar(o,h,l,c,trade_levels["short"]["tp"])
            hit_sl, _ = check_hit_in_bar(o,h,l,c,trade_levels["short"]["sl"])
            if hit_tp:
                short_close_price = trade_levels["short"]["tp"] * (1 - SLIPPAGE_PCT)
                short_exit_type = "tp"
                short_open = False

                long_close_price = short_close_price
                long_exit_type = "oco_due_to_short"
                long_open = False
                break
            elif hit_sl:
                short_close_price = trade_levels["short"]["sl"] * (1 + SLIPPAGE_PCT)
                short_exit_type = "sl"
                short_open = False

                long_close_price = short_close_price
                long_exit_type = "oco_due_to_short"
                long_open = False
                break

        j += 1

    # Close any remaining open at last bar
    final_price = float(df["close"].iloc[-1])
    if long_open:
        long_close_price = final_price * (1 - SLIPPAGE_PCT)
        long_exit_type = "mtm"
    if short_open:
        short_close_price = final_price * (1 + SLIPPAGE_PCT)
        short_exit_type = "mtm"

    long_pl = (long_close_price - entry_price) * size
    short_pl = (entry_price - short_close_price) * size

    # exit fees
    long_exit_fee = abs(long_close_price*size)*TAKER_FEE_RATE
    short_exit_fee = abs(short_close_price*size)*TAKER_FEE_RATE
    total_exit_fees = long_exit_fee + short_exit_fee

    capital += (long_pl + short_pl)
    capital -= total_exit_fees

    # legs stats
    if long_pl > 0:
        legs_stats["long_win"] +=1
    else:
        legs_stats["long_loss"] +=1
    if short_pl >0:
        legs_stats["short_win"] +=1
    else:
        legs_stats["short_loss"] +=1

    trade_num +=1
    results.append({
        "trade": trade_num,
        "entry_index": i,
        "entry_time": df["time"].iloc[i],
        "bias": bias,
        "size": size,
        "long_entry": entry_price,
        "short_entry": entry_price,
        "long_close_price": long_close_price,
        "short_close_price": short_close_price,
        "long_exit_type": long_exit_type,
        "short_exit_type": short_exit_type,
        "long_pl": long_pl,
        "short_pl": short_pl,
        "fees_entry": total_entry_fees,
        "fees_exit": total_exit_fees,
        "net": long_pl + short_pl - total_exit_fees
    })

    equity_curve.append(capital)
    if not ALLOW_OVERLAP:
        i = j
    else:
        i +=1

# -------------------------
# AGGREGATE & REPORT
# -------------------------
res_df = pd.DataFrame(results)
net_pls = res_df["net"].values if not res_df.empty else np.array([])

total_net = capital - INITIAL_CAPITAL
avg_pair = np.mean(net_pls) if len(net_pls)>0 else 0.0
median_pair = np.median(net_pls) if len(net_pls)>0 else 0.0
win_rate = np.mean([1 if x>0 else 0 for x in net_pls])*100 if len(net_pls)>0 else 0.0

equity = np.array(equity_curve)
running_max = np.maximum.accumulate(equity)
drawdowns = (running_max - equity)/running_max
max_dd = np.max(drawdowns) if len(drawdowns)>0 else 0.0

print("===== REALISTIC BACKTEST (Binance Futures VIP0 + RR) =====")
print(f"Pairs traded: {len(res_df)}")
print(f"Starting capital: ${INITIAL_CAPITAL:.2f}")
print(f"Ending capital:  ${capital:.2f}")
print(f"Total Net P/L:   ${total_net:.2f}")
print(f"Average pair P/L: ${avg_pair:.2f}")
print(f"Median pair P/L:  ${median_pair:.2f}")
print(f"Win rate (pair net > 0): {win_rate:.2f}%")
print(f"Long legs: wins={legs_stats['long_win']}, losses={legs_stats['long_loss']}")
print(f"Short legs: wins={legs_stats['short_win']}, losses={legs_stats['short_loss']}")
print(f"Max drawdown: {max_dd*100:.2f}%")

plt.figure(figsize=(10,4))
plt.plot(equity, marker='o', linewidth=1, markersize=2)
plt.title('Equity Curve')
plt.xlabel('Trade number')
plt.ylabel('Capital (USD)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

res_df.to_csv("hedged_binance_results_rr.csv", index=False)
print("Saved detailed trades to hedged_binance_results_rr.csv")
