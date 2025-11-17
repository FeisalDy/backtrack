# hedged_fixed_with_dataset_info.py
import yfinance as yf
import pandas as pd
import numpy as np

# -------------------------
# USER PARAMETERS
# -------------------------
SYMBOL = "BTC-USD"
INTERVAL = "15m"      # supported: 1m,5m,15m,1h,1d
LIMIT = 5760          # ~60 days of 15m candles (60 days * 24 hours * 4 candles/hour)

# Tick-bar analysis (how many rows = one tick-bar)
TICK_SIZE = 1000  # treat each input-row as a "tick" for grouping into 1000-tick bars

# Hedging params (points in quote currency, e.g. USD)
TP_BIG   = 12.0
TP_SMALL = 10.0
SL_BIG   = 12.0
SL_SMALL = 10.0

POSITION_SIZE = 1.0      # units (not $) — profit computed in quote currency per unit
ALLOW_OVERLAP = False    # if False, do not open a new pair until previous pair fully closed

# Market realism
SPREAD = 0.0            # total spread in quote (applied as half to each side on entry)
SLIPPAGE = 0.0          # additional slippage (absolute points) applied on fills
FEE_PER_TRADE = 0.0     # fee per leg (absolute points) — subtract from leg P/L

# -------------------------
# DOWNLOAD DATA (yfinance)
# -------------------------
def get_price(symbol="BTC-USD", interval="15m", limit=5760):
    tf_map = {"1m":"1m","5m":"5m","15m":"15m","1h":"60m","4h":"240m","1d":"1d"}
    interval_yf = tf_map.get(interval, "15m")
    # period chosen to reasonably cover 'limit' candles - using 60d for max 15m data available
    df = yf.download(tickers=symbol, interval=interval_yf, period="60d", auto_adjust=True)
    df = df.reset_index()
    df = df.rename(columns={"Datetime":"time","Date":"time","Open":"open","High":"high","Low":"low","Close":"close"})
    df = df[["time","open","high","low","close"]].tail(limit).reset_index(drop=True)
    return df

df = get_price(SYMBOL, INTERVAL, LIMIT)

# -------------------------
# DATASET INFO (size + span)
# -------------------------
print("===== DATASET INFO =====")
print(f"Rows loaded: {len(df):,}")

# ensure 'time' is datetime
df['time'] = pd.to_datetime(df['time'])

if len(df) > 0:
    start_time = df['time'].iloc[0]
    end_time = df['time'].iloc[-1]
    span = end_time - start_time
    print(f"Start: {start_time}")
    print(f"End:   {end_time}")
    print(f"Total span: {span}")
else:
    print("No rows loaded; exiting.")
    raise SystemExit(1)

# -------------------------
# INDICATORS (example MA bias)
# -------------------------
df["fast_ma"] = df["close"].rolling(20).mean()
df["slow_ma"] = df["close"].rolling(50).mean()

def get_bias(df, i):
    """Return 'bull' or 'bear' or None"""
    fast = df["fast_ma"].iloc[i]
    slow = df["slow_ma"].iloc[i]
    if pd.isna(fast) or pd.isna(slow):
        return None
    return "bull" if fast > slow else "bear"

# -------------------------
# Helpers: intrabar path hit detection
# Assumed path: open -> high -> low -> close
# We check each monotonic subsegment for crossing the target price.
# -------------------------
def check_hit_in_bar(open_p, high_p, low_p, close_p, target):
    """
    Return tuple (hit, hit_price, phase_str) where hit is True if target is reached
    and hit_price is the target (we assume exact TP/SL price used as fill).
    Phase strings: 'open->high', 'high->low', 'low->close' (or None)
    """
    # open -> high (increasing)
    if open_p <= target <= high_p:
        return True, target, "open->high"
    # high -> low (decreasing)
    if low_p <= target <= high_p:
        return True, target, "high->low"
    # low -> close
    if min(low_p, close_p) <= target <= max(low_p, close_p):
        return True, target, "low->close"
    return False, None, None

# -------------------------
# Backtest simulation (fixed engine)
# -------------------------
results = []
legs_stats = {"long_win":0, "long_loss":0, "short_win":0, "short_loss":0}
trades_count = 0

i = 0
while i < len(df):
    bias = get_bias(df, i)
    if bias is None:
        i += 1
        continue

    entry = float(df["close"].iloc[i])

    # Apply spread on entry: long buys at entry + spread/2, short sells at entry - spread/2
    long_entry_price  = entry + (SPREAD/2)
    short_entry_price = entry - (SPREAD/2)

    if bias == "bull":
        long_tp  = long_entry_price + TP_BIG
        long_sl  = long_entry_price - SL_SMALL
        short_tp = short_entry_price - TP_SMALL
        short_sl = short_entry_price + SL_BIG
    else:
        long_tp  = long_entry_price + TP_SMALL
        long_sl  = long_entry_price - SL_BIG
        short_tp = short_entry_price - TP_BIG
        short_sl = short_entry_price + SL_SMALL

    long_open = True
    short_open = True
    long_close_price = None
    short_close_price = None
    long_close_index = None
    short_close_index = None

    # simulate forward bars until both legs closed
    j = i + 1
    while j < len(df) and (long_open or short_open):
        o = float(df["open"].iloc[j])
        h = float(df["high"].iloc[j])
        l = float(df["low"].iloc[j])
        c = float(df["close"].iloc[j])

        # check long leg if still open
        if long_open:
            # first check TP
            hit_tp, hit_price, phase = check_hit_in_bar(o,h,l,c,long_tp)
            if hit_tp:
                fill_price = long_tp + SLIPPAGE
                pl = (fill_price - long_entry_price) * POSITION_SIZE - FEE_PER_TRADE
                long_open = False
                long_close_price = fill_price
                long_close_index = j
                if pl > 0:
                    legs_stats["long_win"] += 1
                else:
                    legs_stats["long_loss"] += 1
            else:
                # check SL
                hit_sl, hit_price, phase = check_hit_in_bar(o,h,l,c,long_sl)
                if hit_sl:
                    fill_price = long_sl - SLIPPAGE
                    pl = (fill_price - long_entry_price) * POSITION_SIZE - FEE_PER_TRADE
                    long_open = False
                    long_close_price = fill_price
                    long_close_index = j
                    if pl > 0:
                        legs_stats["long_win"] += 1
                    else:
                        legs_stats["long_loss"] += 1

        # check short leg if still open
        if short_open:
            hit_tp_s, hit_price_s, phase_s = check_hit_in_bar(o,h,l,c,short_tp)
            if hit_tp_s:
                fill_price = short_tp - SLIPPAGE
                pl_s = (short_entry_price - fill_price) * POSITION_SIZE - FEE_PER_TRADE
                short_open = False
                short_close_price = fill_price
                short_close_index = j
                if pl_s > 0:
                    legs_stats["short_win"] += 1
                else:
                    legs_stats["short_loss"] += 1
            else:
                hit_sl_s, hit_price_s, phase_s = check_hit_in_bar(o,h,l,c,short_sl)
                if hit_sl_s:
                    fill_price = short_sl + SLIPPAGE
                    pl_s = (short_entry_price - fill_price) * POSITION_SIZE - FEE_PER_TRADE
                    short_open = False
                    short_close_price = fill_price
                    short_close_index = j
                    if pl_s > 0:
                        legs_stats["short_win"] += 1
                    else:
                        legs_stats["short_loss"] += 1

        j += 1

    # If a leg never closed before data end, close it at last close price (mark-to-market)
    final_close_price = float(df["close"].iloc[-1])
    if long_open:
        fill_price = final_close_price - SLIPPAGE
        pl = (fill_price - long_entry_price) * POSITION_SIZE - FEE_PER_TRADE
        long_open = False
        long_close_price = fill_price
        long_close_index = len(df)-1
        if pl > 0:
            legs_stats["long_win"] += 1
        else:
            legs_stats["long_loss"] += 1

    if short_open:
        fill_price = final_close_price + SLIPPAGE
        pl_s = (short_entry_price - fill_price) * POSITION_SIZE - FEE_PER_TRADE
        short_open = False
        short_close_price = fill_price
        short_close_index = len(df)-1
        if pl_s > 0:
            legs_stats["short_win"] += 1
        else:
            legs_stats["short_loss"] += 1

    long_pl = (long_close_price - long_entry_price) * POSITION_SIZE - FEE_PER_TRADE
    short_pl = (short_entry_price - short_close_price) * POSITION_SIZE - FEE_PER_TRADE
    net = long_pl + short_pl

    results.append({
        "entry_index": i,
        "entry_time": df["time"].iloc[i],
        "bias": bias,
        "long_entry": long_entry_price,
        "short_entry": short_entry_price,
        "long_close_price": long_close_price,
        "short_close_price": short_close_price,
        "long_close_idx": long_close_index,
        "short_close_idx": short_close_index,
        "long_pl": long_pl,
        "short_pl": short_pl,
        "net": net
    })

    trades_count += 1

    # If overlapping pairs not allowed, jump i to the bar after the later close index to avoid overlapping entries
    if not ALLOW_OVERLAP:
        last_close_idx = max(long_close_index, short_close_index)
        i = max(i+1, last_close_idx + 1)
    else:
        i += 1

# -------------------------
# Aggregate results
# -------------------------
net_pls = [r["net"] for r in results]
long_pls = [r["long_pl"] for r in results]
short_pls = [r["short_pl"] for r in results]

print("\n===== BACKTEST RESULT (fixed engine) =====")
print("Pairs traded:", len(results))
print(f"Net P/L: ${np.sum(net_pls):.2f}")
print(f"Average pair P/L: ${np.mean(net_pls):.2f}")
print(f"Median pair P/L: ${np.median(net_pls):.2f}")
print(f"Long legs: wins={legs_stats['long_win']}, losses={legs_stats['long_loss']}")
print(f"Short legs: wins={legs_stats['short_win']}, losses={legs_stats['short_loss']}")
print(f"Win rate (pair net > 0): {np.mean([1 if r>0 else 0 for r in net_pls])*100:.2f}%")
