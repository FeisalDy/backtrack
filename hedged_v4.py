# hedged_v5.py
# Realistic hedged-pair backtester with filters and ATR-based TP/SL
# Features implemented: A (volatility filter), B (wick filter), D (ATR-based TP/SL), E (symbol selection), G (auto TP widening)
# Binance Futures VIP0 assumptions, 10x leverage (changeable)
# Requirements: yfinance, pandas, numpy, matplotlib
# Run: python hedged_v5.py

import math
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

# -------------------------
# USER PARAMETERS (tweak these)
# -------------------------
SYMBOL_CANDIDATES = ["BTC-USD","ETH-USD","SOL-USD","ADA-USD","DOGE-USD"]
INTERVAL = "15m"
LIMIT = 5760

# ATR / filter params
ATR_PERIOD = 14
ATR_VOL_SPIKE_MULT = 1.5   # require ATR > this * rolling ATR (volatility spike)
WICK_RATIO_THRESHOLD = 0.6 # wick must be >= this fraction of the candle range
BODY_RATIO_MAX = 0.3       # body must be <= this fraction of range (indicates wick-heavy candle)

# Multipliers to convert ATR into TP/SL (you can tune)
TP_BIG_ATR_MULT   = 1.2
TP_SMALL_ATR_MULT = 1.0
SL_BIG_ATR_MULT   = 1.2
SL_SMALL_ATR_MULT = 1.0

# When a volatility spike is detected, widen TP by this multiplier (G)
TP_WIDEN_MULT_ON_SPIKE = 1.5

# Starting capital & sizing
INITIAL_CAPITAL = 20.0
LEVERAGE = 10.0
ALLOW_OVERLAP = False

# Market realism
TAKER_FEE_RATE = 0.0004    # Binance Futures VIP0 taker fee
SLIPPAGE_PCT = 0.0002      # slippage per fill (adverse)
MAINTENANCE_MARGIN_RATE = 0.005

# How many top symbols to trade (by recent ATR)
TOP_K_SYMBOLS = 1

# -------------------------
# Helpers: data download, indicators, intrabar hits
# -------------------------

def download_symbol_data(symbol, interval=INTERVAL, limit=LIMIT):
    tf_map = {"1m":"1m","5m":"5m","15m":"15m","1h":"60m","4h":"240m","1d":"1d"}
    interval_yf = tf_map.get(interval, "15m")
    df = yf.download(tickers=symbol, interval=interval_yf, period="60d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        return None
    df = df.reset_index()
    df = df.rename(columns={"Datetime":"time","Date":"time","Open":"open","High":"high","Low":"low","Close":"close"})
    df = df[["time","open","high","low","close"]].tail(limit).reset_index(drop=True)
    df['time'] = pd.to_datetime(df['time'])
    return df


def atr(df, period=ATR_PERIOD):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# Intrabar path: open -> high -> low -> close
# returns list of (target, phase) in chronological order

def check_hits_in_bar_sequence(open_p, high_p, low_p, close_p, targets):
    hits = []
    # open -> high
    seg_min, seg_max = open_p, high_p
    for t in list(targets):
        if seg_min <= t <= seg_max and not any(math.isclose(t, h[0]) for h in hits):
            hits.append((t, 'open->high'))
    # high -> low
    seg_min, seg_max = low_p, high_p
    for t in list(targets):
        if any(math.isclose(t, h[0]) for h in hits):
            continue
        if seg_min <= t <= seg_max:
            hits.append((t, 'high->low'))
    # low -> close
    seg_min, seg_max = min(low_p, close_p), max(low_p, close_p)
    for t in list(targets):
        if any(math.isclose(t, h[0]) for h in hits):
            continue
        if seg_min <= t <= seg_max:
            hits.append((t, 'low->close'))
    return hits

# -------------------------
# Symbol selection: compute ATR on candidates and pick top K
# -------------------------

print("Selecting symbols by ATR...")
atr_scores = {}
for s in SYMBOL_CANDIDATES:
    df_tmp = download_symbol_data(s)
    if df_tmp is None:
        continue
    tr_atr = atr(df_tmp)
    # Use latest ATR as score
    atr_scores[s] = tr_atr.iloc[-1] if not tr_atr.isna().all() else 0.0

if len(atr_scores) == 0:
    raise SystemExit("No symbol data available. Check symbols and network.")

selected = sorted(atr_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K_SYMBOLS]
SELECTED_SYMBOLS = [x[0] for x in selected]
print("Selected symbols:", SELECTED_SYMBOLS)

# For simplicity run backtest on the first selected symbol only
SYMBOL = SELECTED_SYMBOLS[0]
print(f"Running backtest on {SYMBOL}")

df = download_symbol_data(SYMBOL)
if df is None or df.empty:
    raise SystemExit("Failed to download price data for selected symbol")

# compute ATR and rolling ATR for spike detection
df['atr'] = atr(df)
df['rolling_atr_mean'] = df['atr'].rolling(50).mean()

# moving averages for bias
df['fast_ma'] = df['close'].rolling(20).mean()
df['slow_ma'] = df['close'].rolling(50).mean()

# -------------------------
# Backtest: apply filters, ATR-based TP/SL, widen TP on spike
# -------------------------
capital = INITIAL_CAPITAL
equity = [capital]
results = []
legs_stats = {'long_win':0,'long_loss':0,'short_win':0,'short_loss':0}
trade_n = 0

print('Starting backtest... (filters: ATR spike + wick + ATR-based TP/SL)')

def get_bias(i):
    fast = df['fast_ma'].iloc[i]
    slow = df['slow_ma'].iloc[i]
    if pd.isna(fast) or pd.isna(slow):
        return None
    return 'bull' if fast > slow else 'bear'

i = 0
while i < len(df):
    bias = get_bias(i)
    if bias is None:
        i += 1
        continue

    # filters
    # take current candle info
    entry_time = df['time'].iloc[i]
    entry_price = float(df['close'].iloc[i])
    atr_now = df['atr'].iloc[i]
    atr_mean = df['rolling_atr_mean'].iloc[i]

    # volatility spike filter
    atr_spike = False
    if not pd.isna(atr_now) and not pd.isna(atr_mean) and atr_now > ATR_VOL_SPIKE_MULT * atr_mean:
        atr_spike = True

    # wick filter on next candle (we will use candle i+1 open-high-low-close to detect wick)
    if i+1 >= len(df):
        break
    o = float(df['open'].iloc[i+1])
    h = float(df['high'].iloc[i+1])
    l = float(df['low'].iloc[i+1])
    c = float(df['close'].iloc[i+1])
    rng = h - l if h - l > 1e-12 else 1e-12
    body = abs(c - o)
    upper_wick = h - max(o,c)
    lower_wick = min(o,c) - l
    upper_ratio = upper_wick / rng
    lower_ratio = lower_wick / rng
    body_ratio = body / rng

    wick_signal = False
    # if candle has a pronounced lower wick and small body -> bullish rejection
    if lower_ratio >= WICK_RATIO_THRESHOLD and body_ratio <= BODY_RATIO_MAX:
        wick_signal = True
    # if pronounced upper wick and small body -> bearish rejection
    if upper_ratio >= WICK_RATIO_THRESHOLD and body_ratio <= BODY_RATIO_MAX:
        wick_signal = True

    # Combined filter: require ATR spike AND wick_signal for high-quality setups
    # but be a bit lenient: allow trade if either atr_spike OR wick_signal (tunable)
    if not (atr_spike or wick_signal):
        i += 1
        continue

    # derive TP/SL from ATR (D)
    tp_big = TP_BIG_ATR_MULT * atr_now
    tp_small = TP_SMALL_ATR_MULT * atr_now
    sl_big = SL_BIG_ATR_MULT * atr_now
    sl_small = SL_SMALL_ATR_MULT * atr_now

    # apply automatic widening if spike
    if atr_spike:
        tp_big *= TP_WIDEN_MULT_ON_SPIKE
        tp_small *= TP_WIDEN_MULT_ON_SPIKE

    # ensure minimal TP to avoid too small targets (in USD)
    MIN_TP = 1.0
    tp_big = max(tp_big, MIN_TP)
    tp_small = max(tp_small, MIN_TP)
    sl_big = max(sl_big, 1.0)
    sl_small = max(sl_small, 1.0)

    # sizing
    size = (capital * LEVERAGE) / entry_price
    if size <= 0:
        i += 1
        continue

    # simple margin check
    notional = entry_price * size
    init_margin = notional / LEVERAGE
    total_init_margin = init_margin * 2
    est_entry_fees = 2 * (notional * TAKER_FEE_RATE)
    if capital - est_entry_fees <= total_init_margin * MAINTENANCE_MARGIN_RATE:
        i += 1
        continue

    # set TP/SL depending on bias (as earlier scheme)
    if bias == 'bull':
        long_tp = entry_price + tp_big
        long_sl = entry_price - sl_small
        short_tp = entry_price - tp_small
        short_sl = entry_price + sl_big
    else:
        long_tp = entry_price + tp_small
        long_sl = entry_price - sl_big
        short_tp = entry_price - tp_big
        short_sl = entry_price + sl_small

    # charge entry fees
    entry_notional = entry_price * size
    fee_entry_leg = entry_notional * TAKER_FEE_RATE
    total_entry_fees = 2 * fee_entry_leg
    capital -= total_entry_fees

    # simulate forward bars until both legs closed
    long_open = True
    short_open = True
    long_close_price = None
    short_close_price = None
    long_exit_type = None
    short_exit_type = None
    long_close_idx = None
    short_close_idx = None

    j = i + 1
    while j < len(df) and (long_open or short_open):
        o = float(df['open'].iloc[j])
        h = float(df['high'].iloc[j])
        l = float(df['low'].iloc[j])
        c = float(df['close'].iloc[j])

        # collect targets only for legs still open
        targets = []
        if long_open:
            targets.extend([long_tp, long_sl])
        if short_open:
            targets.extend([short_tp, short_sl])

        hits = check_hits_in_bar_sequence(o,h,l,c,targets)

        # process hits in chronological order; DO NOT auto-close opposite leg
        for price_hit, phase in hits:
            if long_open and (math.isclose(price_hit,long_tp) or math.isclose(price_hit,long_sl)):
                if math.isclose(price_hit,long_tp):
                    long_close_price = long_tp * (1 + SLIPPAGE_PCT)
                    long_exit_type = 'tp'
                else:
                    long_close_price = long_sl * (1 - SLIPPAGE_PCT)
                    long_exit_type = 'sl'
                long_open = False
                long_close_idx = j

            if short_open and (math.isclose(price_hit,short_tp) or math.isclose(price_hit,short_sl)):
                if math.isclose(price_hit,short_tp):
                    short_close_price = short_tp * (1 - SLIPPAGE_PCT)
                    short_exit_type = 'tp'
                else:
                    short_close_price = short_sl * (1 + SLIPPAGE_PCT)
                    short_exit_type = 'sl'
                short_open = False
                short_close_idx = j

        # liquidation check using mark price = close
        mark = c
        unreal_long = (mark - entry_price) * size if long_open else 0.0
        unreal_short = (entry_price - mark) * size if short_open else 0.0
        unreal = unreal_long + unreal_short
        margin_used = (entry_notional / LEVERAGE) * 2
        eq_for_liq = capital + unreal
        maintenance = margin_used * MAINTENANCE_MARGIN_RATE
        if eq_for_liq <= maintenance:
            # liquidate open legs at worst prices
            if long_open:
                long_close_price = l * (1 - SLIPPAGE_PCT)
                long_exit_type = 'liquidation'
                long_open = False
                long_close_idx = j
            if short_open:
                short_close_price = h * (1 + SLIPPAGE_PCT)
                short_exit_type = 'liquidation'
                short_open = False
                short_close_idx = j
            break

        if (not long_open) and (not short_open):
            break

        j += 1

    # close any remaining open legs at final
    final_price = float(df['close'].iloc[-1])
    if long_open:
        long_close_price = final_price * (1 - SLIPPAGE_PCT)
        long_exit_type = 'mtm'
        long_close_idx = len(df)-1
    if short_open:
        short_close_price = final_price * (1 + SLIPPAGE_PCT)
        short_exit_type = 'mtm'
        short_close_idx = len(df)-1

    # realized pnl per leg
    long_pl = (long_close_price - entry_price) * size
    short_pl = (entry_price - short_close_price) * size

    # exit fees
    long_exit_fee = abs(long_close_price * size) * TAKER_FEE_RATE
    short_exit_fee = abs(short_close_price * size) * TAKER_FEE_RATE
    total_exit_fees = long_exit_fee + short_exit_fee

    capital += (long_pl + short_pl)
    capital -= total_exit_fees

    # update stats
    if long_pl > 0:
        legs_stats['long_win'] += 1
    else:
        legs_stats['long_loss'] += 1
    if short_pl > 0:
        legs_stats['short_win'] += 1
    else:
        legs_stats['short_loss'] += 1

    trade_n += 1
    results.append({'trade':trade_n,'entry_time':entry_time,'bias':bias,'size':size,'long_entry':entry_price,'short_entry':entry_price,
                    'long_close_price':long_close_price,'short_close_price':short_close_price,'long_exit_type':long_exit_type,
                    'short_exit_type':short_exit_type,'long_pl':long_pl,'short_pl':short_pl,'fees_entry':total_entry_fees,
                    'fees_exit':total_exit_fees,'net':long_pl+short_pl-total_exit_fees})

    equity.append(capital)

    # move index according to overlap rule
    if not ALLOW_OVERLAP:
        last_close_idx = max(long_close_idx or 0, short_close_idx or 0)
        i = max(i+1, last_close_idx + 1)
    else:
        i += 1

# -------------------------
# Aggregate results & output
# -------------------------
res = pd.DataFrame(results)
net_pls = res['net'].values if not res.empty else np.array([])

total_net = capital - INITIAL_CAPITAL
avg = np.mean(net_pls) if len(net_pls)>0 else 0.0
med = np.median(net_pls) if len(net_pls)>0 else 0.0
win_rate = np.mean([1 if x>0 else 0 for x in net_pls])*100 if len(net_pls)>0 else 0.0

equity_arr = np.array(equity)
running_max = np.maximum.accumulate(equity_arr)
drawdowns = (running_max - equity_arr) / running_max
max_dd = np.max(drawdowns) if len(drawdowns)>0 else 0.0

print('===== HEDGED_V5 BACKTEST (filters + ATR TP/SL + TP widen) =====')
print(f'Pairs traded: {len(res)}')
print(f'Starting capital: ${INITIAL_CAPITAL:.2f}')
print(f'Ending capital:  ${capital:.2f}')
print(f'Total Net P/L:   ${total_net:.2f}')
print(f'Average pair P/L: ${avg:.2f}')
print(f'Median pair P/L:  ${med:.2f}')
print(f'Win rate (pair net > 0): {win_rate:.2f}%')
print(f"Long legs: wins={legs_stats['long_win']}, losses={legs_stats['long_loss']}")
print(f"Short legs: wins={legs_stats['short_win']}, losses={legs_stats['short_loss']}")
print(f'Max drawdown: {max_dd*100:.2f}%')

# plot equity
plt.figure(figsize=(10,4))
plt.plot(equity_arr, marker='o', linewidth=1, markersize=2)
plt.title('Equity Curve (hedged_v5)')
plt.xlabel('Trade number')
plt.ylabel('Capital (USD)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# save
res.to_csv('hedged_v5_results.csv', index=False)
print('Saved hedged_v5_results.csv')