# ===== IMPROVED BACKTEST RESULTS =====
# Trades executed: 41
# Starting capital: $20.00
# Ending capital:  $38.48
# Total Net P/L:   $18.48
# Average trade P/L: $0.49
# Median trade P/L:  $0.55
# Win rate: 53.66%
# Legs: wins=22, losses=19
# Max drawdown: 11.44%

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
INTERVAL = "30m"
LIMIT = 5760

INITIAL_CAPITAL = 20.0
LEVERAGE = 3.0  # reduced for stability

TAKER_FEE_RATE = 0.0004
SLIPPAGE_PCT = 0.0002

RR_FOLLOW = 2.5  # Increased RR ratio for better risk/reward
ATR_SL_MULT = 2.0  # Wider stops to avoid premature exits

ALLOW_OVERLAP = False

# -------------------------
# DATA DOWNLOAD
# -------------------------
def get_price(symbol="BTC-USD", interval="15m", limit=5760):
    tf_map = {"1m":"1m","5m":"5m","15m":"15m","30m":"30m","1h":"1h","4h":"4h","1d":"1d"}
    interval_yf = tf_map.get(interval, "15m")
    df = yf.download(
        tickers=symbol,
        interval=interval_yf,
        period="60d",
        auto_adjust=False,
        progress=False
    )
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.reset_index()
    df = df.rename(columns={"Datetime":"time","Date":"time","Open":"open","High":"high",
                            "Low":"low","Low":"low","Close":"close","Volume":"volume"})
    if "volume" not in df.columns:
        df["volume"] = 1
    df = df[["time","open","high","low","close","volume"]].tail(limit).reset_index(drop=True)
    df['time'] = pd.to_datetime(df['time'])
    return df

df = get_price(SYMBOL, INTERVAL, LIMIT)
if df.empty:
    raise SystemExit("No data downloaded. Check connectivity and ticker.")

# -------------------------
# INDICATORS
# -------------------------
def compute_indicators(df):
    # Original indicators
    df["fast_ma"] = df["close"].rolling(20).mean()
    df["slow_ma"] = df["close"].rolling(50).mean()
    df["medium_ma"] = df["close"].rolling(100).mean()  # Multi-timeframe
    
    # OBV
    df["delta_close"] = df["close"].diff()
    df["obv"] = 0
    for i in range(1, len(df)):
        if df["delta_close"].iloc[i] > 0:
            df["obv"].iloc[i] = df["obv"].iloc[i-1] + df["volume"].iloc[i]
        elif df["delta_close"].iloc[i] < 0:
            df["obv"].iloc[i] = df["obv"].iloc[i-1] - df["volume"].iloc[i]
        else:
            df["obv"].iloc[i] = df["obv"].iloc[i-1]
    
    # ATR
    df["tr"] = df["high"] - df["low"]
    df["atr"] = df["tr"].rolling(14).mean()
    df["atr_median"] = df["atr"].rolling(50).median()  # Trend strength filter
    
    # Improved indicators with better horizons
    # 1. Price Momentum (multiple timeframes)
    df['price_momentum_3'] = (df['close'] - df['close'].shift(3)) / df['close'].shift(3) * 100
    df['price_momentum_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100
    
    # 2. Recent High/Low Breakout (multiple windows)
    df['recent_high_5'] = df['high'].rolling(5).max()
    df['recent_low_5'] = df['low'].rolling(5).min()
    df['recent_high_20'] = df['high'].rolling(20).max()  # Medium-term
    df['recent_low_20'] = df['low'].rolling(20).min()
    
    # 3. Volume Surge (current volume vs longer period)
    volume_avg_10 = df['volume'].rolling(10).mean()
    df['volume_surge'] = df['volume'] / volume_avg_10
    
    # 4. Price vs Recent Range (normalized)
    df['price_in_range_5'] = (df['close'] - df['recent_low_5']) / (df['recent_high_5'] - df['recent_low_5'])
    df['price_in_range_20'] = (df['close'] - df['recent_low_20']) / (df['recent_high_20'] - df['recent_low_20'])
    
    # 5. OBV Momentum (longer period - 10-14 bars)
    df['obv_momentum_10'] = df['obv'].diff(10)
    df['obv_momentum_14'] = df['obv'].diff(14)
    
    # 6. Calculate rolling statistics for z-score normalization
    for col in ['price_momentum_3', 'price_momentum_10', 'volume_surge', 'obv_momentum_10', 'obv_momentum_14']:
        if col in df.columns:
            rolling_mean = df[col].rolling(50).mean()
            rolling_std = df[col].rolling(50).std()
            df[f'{col}_zscore'] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
    
    # 7. Trend strength (distance between MAs normalized by ATR)
    df['trend_strength'] = (df['fast_ma'] - df['slow_ma']) / (df['atr'] + 1e-8)
    
    return df

df = compute_indicators(df)

# -------------------------
# IMPROVED BIAS FUNCTION WITH PROBABILISTIC SCORING
# -------------------------
def get_bias(df, i):
    if i < 100:  # Need enough data for indicators and statistics
        return None

    # Check for NaN values in required columns
    required_cols = ['price_in_range_5', 'price_in_range_20', 'recent_high_5', 'recent_low_5',
                    'recent_high_20', 'recent_low_20', 'volume_surge', 'atr', 'atr_median',
                    'price_momentum_3_zscore', 'price_momentum_10_zscore', 'volume_surge_zscore',
                    'obv_momentum_10_zscore', 'obv_momentum_14_zscore', 'trend_strength',
                    'fast_ma', 'slow_ma', 'medium_ma']
    
    for col in required_cols:
        if pd.isna(df[col].iloc[i]):
            return None

    # TREND STRENGTH FILTER: Only trade when ATR > median (volatile enough)
    if df['atr'].iloc[i] <= df['atr_median'].iloc[i]:
        return None

    # Initialize probabilistic score (normalized signals)
    score = 0.0
    
    # 1. SHORT-TERM PRICE POSITION vs RECENT RANGE (z-score weighted)
    price_range_5 = df['price_in_range_5'].iloc[i]
    if price_range_5 > 0.75:  # Strong position at top
        score += 2.5
    elif price_range_5 < 0.25:  # Strong position at bottom
        score -= 2.5
    elif price_range_5 > 0.6:
        score += 1.5
    elif price_range_5 < 0.4:
        score -= 1.5

    # 2. MULTI-TIMEFRAME BREAKOUT CONFIRMATION
    short_breakout_bull = df['close'].iloc[i] > df['recent_high_5'].iloc[i-1]
    short_breakout_bear = df['close'].iloc[i] < df['recent_low_5'].iloc[i-1]
    medium_trend_bull = df['close'].iloc[i] > df['recent_high_20'].iloc[i-1] * 0.995  # Near or above
    medium_trend_bear = df['close'].iloc[i] < df['recent_low_20'].iloc[i-1] * 1.005  # Near or below
    
    # Strong signal: both timeframes align
    if short_breakout_bull and medium_trend_bull:
        score += 4.0
    elif short_breakout_bear and medium_trend_bear:
        score -= 4.0
    # Weak signal: only short-term
    elif short_breakout_bull:
        score += 1.5
    elif short_breakout_bear:
        score -= 1.5

    # 3. PRICE MOMENTUM (z-score based, both timeframes)
    momentum_3_z = df['price_momentum_3_zscore'].iloc[i]
    momentum_10_z = df['price_momentum_10_zscore'].iloc[i]
    
    # Strong momentum alignment
    if momentum_3_z > 1.0 and momentum_10_z > 0.5:  # Both positive, short-term strong
        score += 3.0
    elif momentum_3_z < -1.0 and momentum_10_z < -0.5:  # Both negative
        score -= 3.0
    # Moderate momentum
    elif momentum_3_z > 0.5:
        score += 1.5
    elif momentum_3_z < -0.5:
        score -= 1.5

    # 4. VOLUME CONFIRMATION (z-score based)
    volume_z = df['volume_surge_zscore'].iloc[i]
    price_up = df['price_momentum_3_zscore'].iloc[i] > 0.5
    price_down = df['price_momentum_3_zscore'].iloc[i] < -0.5
    
    # High volume surge (z-score > 1.5 means significantly above normal)
    if volume_z > 1.5:
        if price_up:
            score += 2.5
        elif price_down:
            score -= 2.5
    elif volume_z > 1.0:
        if price_up:
            score += 1.5
        elif price_down:
            score -= 1.5

    # 5. OBV MOMENTUM (longer period, z-score based)
    obv_10_z = df['obv_momentum_10_zscore'].iloc[i]
    obv_14_z = df['obv_momentum_14_zscore'].iloc[i]
    
    # Both OBV timeframes agree
    if obv_10_z > 0.5 and obv_14_z > 0.5:
        score += 2.0
    elif obv_10_z < -0.5 and obv_14_z < -0.5:
        score -= 2.0
    elif obv_10_z > 1.0 or obv_14_z > 1.0:  # At least one is strong
        score += 1.0
    elif obv_10_z < -1.0 or obv_14_z < -1.0:
        score -= 1.0

    # 6. TREND STRENGTH (normalized by ATR)
    trend_str = df['trend_strength'].iloc[i]
    if abs(trend_str) > 2.0:  # Strong trend
        if trend_str > 0:
            score += 2.0
        else:
            score -= 2.0
    elif abs(trend_str) > 1.0:  # Moderate trend
        if trend_str > 0:
            score += 1.0
        else:
            score -= 1.0

    # 7. MULTI-TIMEFRAME MA CONFIRMATION
    if df['fast_ma'].iloc[i] > df['slow_ma'].iloc[i] > df['medium_ma'].iloc[i]:
        score += 1.5  # Strong uptrend
    elif df['fast_ma'].iloc[i] < df['slow_ma'].iloc[i] < df['medium_ma'].iloc[i]:
        score -= 1.5  # Strong downtrend

    # BIAS DETERMINATION (adjusted thresholds for probabilistic scoring)
    if score >= 8.0:  # Strong bullish signal
        return "bull"
    elif score <= -8.0:  # Strong bearish signal
        return "bear"
    else:
        return None

# -------------------------
# ATR-BASED TRADE LEVELS
# -------------------------
def calculate_trade_levels(entry_price, atr_val, bias, rr_follow=2.0, atr_sl_mult=1.5):
    if bias == "bull":
        stop = entry_price - atr_val * atr_sl_mult
        risk = entry_price - stop
        tp = entry_price + risk * rr_follow
    else:
        stop = entry_price + atr_val * atr_sl_mult
        risk = stop - entry_price
        tp = entry_price - risk * rr_follow
    return {"tp": tp, "sl": stop}

# -------------------------
# CHECK SL/TP HIT
# -------------------------
def check_stops_in_bar(o, h, l, c, tp_level, sl_level, bias):
    hit_tp = False
    hit_sl = False
    if bias == "bull":
        if h >= tp_level: hit_tp = True
        if l <= sl_level: hit_sl = True
    else:
        if l <= tp_level: hit_tp = True
        if h >= sl_level: hit_sl = True
    if hit_tp and hit_sl:
        return False, True  # conservative: assume SL hit first
    return hit_tp, hit_sl

# -------------------------
# BACKTEST
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
    atr_val = float(df["atr"].iloc[i])
    levels = calculate_trade_levels(entry_price, atr_val, bias, RR_FOLLOW, ATR_SL_MULT)

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

        if hit_tp:
            close_price = levels["tp"] * ((1 + SLIPPAGE_PCT) if bias=="bull" else (1 - SLIPPAGE_PCT))
            exit_type = "tp"
            trade_open = False
            break
        elif hit_sl:
            close_price = levels["sl"] * ((1 - SLIPPAGE_PCT) if bias=="bull" else (1 + SLIPPAGE_PCT))
            exit_type = "sl"
            trade_open = False
            break
        j += 1

    if trade_open:
        final_price = float(df["close"].iloc[-1])
        close_price = final_price * ((1 + SLIPPAGE_PCT) if bias=="bull" else (1 - SLIPPAGE_PCT))
        exit_type = "mtm"

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

print("\n===== IMPROVED BACKTEST RESULTS =====")
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
plt.title('Equity Curve (Improved ATR + Bias Logic)')
plt.xlabel('Trade number')
plt.ylabel('Capital (USD)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

res_df.to_csv("bias_only_improved.csv", index=False)
