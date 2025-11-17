# ===== CRYPTO TRADING BOT BACKTEST V9 (Enhanced with Advanced Filters) =====
# Implements:
# - Tighter signal thresholds (score >= 9 for bull, <= -9 for bear)
# - Dynamic RR and SL based on volatility and trend strength
# - Pre-entry filters (volume, sideways markets, momentum alignment)
# - OBV/Volume alignment requirement
# - Cooldown period after losses
# - Higher timeframe confirmation

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yfinance as yf

# -------------------------
# USER PARAMETERS
# -------------------------
SYMBOL = "BTC-USD"  # BTC-USD, ETH-USD, DOGE-USD, SOL-USD, ADA-USD
INTERVAL = "30m"      # "5m", "15m", "30m", "1h"
LIMIT = 10000         # Max bars to use for backtest

INITIAL_CAPITAL = 20.0
LEVERAGE = 2

TAKER_FEE_RATE = 0.0004
SLIPPAGE_PCT = 0.0002

# Dynamic parameters (will be adjusted based on conditions)
BASE_RR_FOLLOW = 2.5
BASE_ATR_SL_MULT = 2.0

ALLOW_OVERLAP = False

# New filter parameters
MIN_SCORE_THRESHOLD = 9.0  # Tightened from 8.0
MIN_TREND_STRENGTH = 0.5   # Avoid sideways markets
COOLDOWN_BARS = 10          # Bars to wait after a loss
REQUIRE_OBV_VOLUME_ALIGN = True  # Both OBV and volume must align

SCORE_MIN = 9.0
SCORE_MAX = 11.0  # Only trade in this range
# -------------------------
# DATA LOADING
# -------------------------
def load_from_yfinance(symbol, interval, limit):
    print(f"Fetching {symbol} {interval} data from Yahoo Finance...")
    df = yf.download(symbol, period="max", interval=interval)
    
    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].rename(
        columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'}
    )
    df = df.reset_index()
    df['time'] = pd.to_datetime(df['Datetime'] if 'Datetime' in df.columns else df['Date'])
    df = df.drop(columns=['Datetime'] if 'Datetime' in df.columns else ['Date'])
    if len(df) > limit:
        df = df.tail(limit).reset_index(drop=True)
        print(f"Using last {limit:,} bars for backtest")
    print(f"Loaded {len(df):,} bars | {df['time'].iloc[0].strftime('%Y-%m-%d')} to {df['time'].iloc[-1].strftime('%Y-%m-%d')}")
    return df

# Load data
df = load_from_yfinance(SYMBOL, INTERVAL, LIMIT)

# -------------------------
# INDICATORS
# -------------------------
def compute_indicators(df):
    df["fast_ma"] = df["close"].rolling(20).mean()
    df["slow_ma"] = df["close"].rolling(50).mean()
    df["medium_ma"] = df["close"].rolling(100).mean()
    
    # Higher timeframe trend approximation (using longer MAs)
    df["htf_fast"] = df["close"].rolling(100).mean()  # ~4h equivalent for 30m
    df["htf_slow"] = df["close"].rolling(200).mean()  # ~8h equivalent for 30m
    
    # OBV
    df["delta_close"] = df["close"].diff()
    df["obv"] = 0.0
    for i in range(1, len(df)):
        if df["delta_close"].iloc[i] > 0:
            df.loc[df.index[i], "obv"] = df["obv"].iloc[i-1] + df["volume"].iloc[i]
        elif df["delta_close"].iloc[i] < 0:
            df.loc[df.index[i], "obv"] = df["obv"].iloc[i-1] - df["volume"].iloc[i]
        else:
            df.loc[df.index[i], "obv"] = df["obv"].iloc[i-1]
    
    # ATR
    df["tr"] = df["high"] - df["low"]
    df["atr"] = df["tr"].rolling(14).mean()
    df["atr_median"] = df["atr"].rolling(50).median()
    df["atr_percentile"] = df["atr"].rolling(100).apply(lambda x: (x.iloc[-1] <= x).sum() / len(x) * 100)
    
    # Price Momentum
    df['price_momentum_3'] = (df['close'] - df['close'].shift(3)) / df['close'].shift(3) * 100
    df['price_momentum_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100
    
    # Recent High/Low Breakout
    df['recent_high_5'] = df['high'].rolling(5).max()
    df['recent_low_5'] = df['low'].rolling(5).min()
    df['recent_high_20'] = df['high'].rolling(20).max()
    df['recent_low_20'] = df['low'].rolling(20).min()
    
    # Volume metrics
    df['volume_avg_10'] = df['volume'].rolling(10).mean()
    df['volume_avg_50'] = df['volume'].rolling(50).mean()
    df['volume_surge'] = df['volume'] / df['volume_avg_10']
    
    # Price vs Recent Range
    df['price_in_range_5'] = (df['close'] - df['recent_low_5']) / (df['recent_high_5'] - df['recent_low_5'] + 1e-8)
    df['price_in_range_20'] = (df['close'] - df['recent_low_20']) / (df['recent_high_20'] - df['recent_low_20'] + 1e-8)
    
    # OBV Momentum
    df['obv_momentum_10'] = df['obv'].diff(10)
    df['obv_momentum_14'] = df['obv'].diff(14)
    
    # Z-score normalization
    for col in ['price_momentum_3', 'price_momentum_10', 'volume_surge', 'obv_momentum_10', 'obv_momentum_14']:
        rolling_mean = df[col].rolling(50).mean()
        rolling_std = df[col].rolling(50).std()
        df[f'{col}_zscore'] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
    
    # Trend strength
    df['trend_strength'] = (df['fast_ma'] - df['slow_ma']) / (df['atr'] + 1e-8)
    df['htf_trend_strength'] = (df['htf_fast'] - df['htf_slow']) / (df['atr'] + 1e-8)
    
    return df

df = compute_indicators(df)

# -------------------------
# ENHANCED BIAS FUNCTION WITH FILTERS
# -------------------------
def get_bias_and_score(df, i, last_loss_index=None):
    """
    Returns (bias, score, details) tuple
    bias: "bull", "bear", or None
    score: numerical score
    details: dict with filter results for analysis
    """
    if i < 200:  # Need more history for HTF indicators
        return None, 0.0, {}

    details = {
        "filter_nan": False,
        "filter_low_volume": False,
        "filter_sideways": False,
        "filter_weak_atr": False,
        "filter_htf_conflict": False,
        "filter_obv_volume_conflict": False,
        "filter_cooldown": False,
        "filter_momentum_conflict": False,
    }

    # Check for NaN values
    required_cols = ['price_in_range_5', 'price_in_range_20', 'recent_high_5', 'recent_low_5',
                    'recent_high_20', 'recent_low_20', 'volume_surge', 'atr', 'atr_median',
                    'price_momentum_3_zscore', 'price_momentum_10_zscore', 'volume_surge_zscore',
                    'obv_momentum_10_zscore', 'obv_momentum_14_zscore', 'trend_strength',
                    'fast_ma', 'slow_ma', 'medium_ma', 'htf_fast', 'htf_slow', 'volume_avg_50',
                    'htf_trend_strength', 'atr_percentile']
    
    for col in required_cols:
        if pd.isna(df[col].iloc[i]):
            details["filter_nan"] = True
            return None, 0.0, details

    # PRE-FILTERS
    
    # 1. Cooldown after loss
    if last_loss_index is not None and (i - last_loss_index) < COOLDOWN_BARS:
        details["filter_cooldown"] = True
        return None, 0.0, details
    
    # 2. Volume filter - skip low volume periods
    if df['volume'].iloc[i] < df['volume_avg_50'].iloc[i]:
        details["filter_low_volume"] = True
        return None, 0.0, details
    
    # 3. Sideways market filter
    if abs(df['trend_strength'].iloc[i]) < MIN_TREND_STRENGTH:
        details["filter_sideways"] = True
        return None, 0.0, details
    
    # 4. Only trade when ATR > median (volatile enough) but not extreme
    if df['atr'].iloc[i] <= df['atr_median'].iloc[i]:
        details["filter_weak_atr"] = True
        return None, 0.0, details

    # 5. Higher timeframe confirmation
    htf_bullish = df['htf_fast'].iloc[i] > df['htf_slow'].iloc[i]
    htf_bearish = df['htf_fast'].iloc[i] < df['htf_slow'].iloc[i]
    
    # Calculate score
    score = 0.0
    
    # 1. Price position in range (reduced weight to avoid overlap with breakout)
    price_range_5 = df['price_in_range_5'].iloc[i]
    if price_range_5 > 0.75:
        score += 1.5  # Reduced from 2.5
    elif price_range_5 < 0.25:
        score -= 1.5
    elif price_range_5 > 0.6:
        score += 0.8
    elif price_range_5 < 0.4:
        score -= 0.8

    # 2. Multi-timeframe breakout (primary signal)
    short_breakout_bull = df['close'].iloc[i] > df['recent_high_5'].iloc[i-1]
    short_breakout_bear = df['close'].iloc[i] < df['recent_low_5'].iloc[i-1]
    medium_trend_bull = df['close'].iloc[i] > df['recent_high_20'].iloc[i-1] * 0.995
    medium_trend_bear = df['close'].iloc[i] < df['recent_low_20'].iloc[i-1] * 1.005
    
    if short_breakout_bull and medium_trend_bull:
        score += 5.0  # Increased importance
    elif short_breakout_bear and medium_trend_bear:
        score -= 5.0
    elif short_breakout_bull:
        score += 2.0
    elif short_breakout_bear:
        score -= 2.0

    # 3. Price momentum (must align, not just exist)
    momentum_3_z = df['price_momentum_3_zscore'].iloc[i]
    momentum_10_z = df['price_momentum_10_zscore'].iloc[i]
    
    # Strong aligned momentum
    if momentum_3_z > 1.0 and momentum_10_z > 0.5:
        score += 3.5
    elif momentum_3_z < -1.0 and momentum_10_z < -0.5:
        score -= 3.5
    # Weak or conflicting momentum gets less weight
    elif momentum_3_z > 0.8 and momentum_10_z > 0:
        score += 1.5
    elif momentum_3_z < -0.8 and momentum_10_z < 0:
        score -= 1.5

    # 4. Volume confirmation (aligned with price move)
    volume_z = df['volume_surge_zscore'].iloc[i]
    price_up_strong = momentum_3_z > 0.8
    price_down_strong = momentum_3_z < -0.8
    
    if volume_z > 1.5:
        if price_up_strong:
            score += 3.0
        elif price_down_strong:
            score -= 3.0
    elif volume_z > 1.0:
        if price_up_strong:
            score += 1.5
        elif price_down_strong:
            score -= 1.5

    # 5. OBV momentum (must align with volume and price)
    obv_10_z = df['obv_momentum_10_zscore'].iloc[i]
    obv_14_z = df['obv_momentum_14_zscore'].iloc[i]
    
    obv_bullish = obv_10_z > 0.5 and obv_14_z > 0.5
    obv_bearish = obv_10_z < -0.5 and obv_14_z < -0.5
    
    if obv_bullish:
        score += 2.5
    elif obv_bearish:
        score -= 2.5
    elif obv_10_z > 1.0 or obv_14_z > 1.0:
        score += 1.0
    elif obv_10_z < -1.0 or obv_14_z < -1.0:
        score -= 1.0

    # 6. Trend strength
    trend_str = df['trend_strength'].iloc[i]
    if abs(trend_str) > 2.0:
        if trend_str > 0:
            score += 2.5
        else:
            score -= 2.5
    elif abs(trend_str) > 1.0:
        if trend_str > 0:
            score += 1.2
        else:
            score -= 1.2

    # 7. MA confirmation
    if df['fast_ma'].iloc[i] > df['slow_ma'].iloc[i] > df['medium_ma'].iloc[i]:
        score += 2.0
    elif df['fast_ma'].iloc[i] < df['slow_ma'].iloc[i] < df['medium_ma'].iloc[i]:
        score -= 2.0

    # Determine bias direction from score
    potential_bias = None
    if score >= MIN_SCORE_THRESHOLD:
        potential_bias = "bull"
    elif score <= -MIN_SCORE_THRESHOLD:
        potential_bias = "bear"
    else:
        return None, score, details

    # ADDITIONAL CONFLICT CHECKS
    
    # Check HTF alignment
    if potential_bias == "bull" and not htf_bullish:
        details["filter_htf_conflict"] = True
        return None, score, details
    elif potential_bias == "bear" and not htf_bearish:
        details["filter_htf_conflict"] = True
        return None, score, details
    
    # Check OBV/Volume alignment requirement
    if REQUIRE_OBV_VOLUME_ALIGN:
        if potential_bias == "bull":
            if not (obv_bullish and volume_z > 0.5):
                details["filter_obv_volume_conflict"] = True
                return None, score, details
        elif potential_bias == "bear":
            if not (obv_bearish and volume_z > 0.5):
                details["filter_obv_volume_conflict"] = True
                return None, score, details
    
    # Check momentum alignment
    if potential_bias == "bull":
        if momentum_3_z < 0 or momentum_10_z < -0.5:
            details["filter_momentum_conflict"] = True
            return None, score, details
    elif potential_bias == "bear":
        if momentum_3_z > 0 or momentum_10_z > 0.5:
            details["filter_momentum_conflict"] = True
            return None, score, details

    return potential_bias, score, details

# -------------------------
# DYNAMIC TRADE LEVELS
# -------------------------
def calculate_dynamic_trade_levels(df, i, entry_price, bias):
    """
    Calculate adaptive RR and SL based on volatility and trend strength
    """
    atr_val = float(df['atr'].iloc[i])
    trend_str = df['trend_strength'].iloc[i]
    atr_pct = df['atr_percentile'].iloc[i]
    
    # Adaptive SL multiplier
    # Tighter SL when trend is weak, wider when strong
    if abs(trend_str) > 2.0:
        atr_sl_mult = 2.0  # Strong trend, can use wider SL
    elif abs(trend_str) > 1.0:
        atr_sl_mult = 1.7
    else:
        atr_sl_mult = 1.5  # Weak trend, tighter SL
    
    # Adaptive RR ratio
    # Low volatility → higher RR (can capture more)
    # High volatility → lower RR (take profits faster)
    if atr_pct < 30:  # Low volatility
        rr_follow = 3.0
    elif atr_pct < 50:
        rr_follow = 2.5
    elif atr_pct < 70:
        rr_follow = 2.0
    else:  # High volatility
        rr_follow = 1.8
    
    if bias == "bull":
        stop = entry_price - atr_val * atr_sl_mult
        risk = entry_price - stop
        tp = entry_price + risk * rr_follow
    else:
        stop = entry_price + atr_val * atr_sl_mult
        risk = stop - entry_price
        tp = entry_price - risk * rr_follow
    
    return {
        "tp": tp, 
        "sl": stop,
        "atr_sl_mult": atr_sl_mult,
        "rr_follow": rr_follow
    }

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
filter_stats = {
    "total_bars": 0,
    "filter_nan": 0,
    "filter_low_volume": 0,
    "filter_sideways": 0,
    "filter_weak_atr": 0,
    "filter_htf_conflict": 0,
    "filter_obv_volume_conflict": 0,
    "filter_cooldown": 0,
    "filter_momentum_conflict": 0,
    "weak_score": 0,
}
legs_stats = {"win":0, "loss":0}
trade_num = 0
last_loss_index = None

i = 1
pbar = tqdm(total=len(df), desc="Backtesting", unit="bar")
while i < len(df)-1:
    pbar.update(1)
    filter_stats["total_bars"] += 1
    
    bias, score, details = get_bias_and_score(df, i, last_loss_index)
    
    # Track filter statistics
    for key, val in details.items():
        if val:
            filter_stats[key] += 1
    
    if bias is None:
        if abs(score) > 0 and abs(score) < MIN_SCORE_THRESHOLD:
            filter_stats["weak_score"] += 1
        i += 1
        continue

    entry_price = float(df["open"].iloc[i+1])
    levels = calculate_dynamic_trade_levels(df, i, entry_price, bias)

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

    net = pl - exit_fee
    
    if net > 0:
        legs_stats["win"] += 1
        last_loss_index = None  # Reset cooldown on win
    else:
        legs_stats["loss"] += 1
        last_loss_index = i  # Start cooldown

    trade_num += 1
    results.append({
        "trade": trade_num,
        "entry_index": i+1,
        "entry_time": df["time"].iloc[i+1],
        "bias": bias,
        "score": score,
        "size": size,
        "entry": entry_price,
        "stop_loss": levels['sl'],
        "take_profit": levels['tp'],
        "atr_sl_mult": levels['atr_sl_mult'],
        "rr_ratio": levels['rr_follow'],
        "close_price": close_price,
        "exit_type": exit_type,
        "pl": pl,
        "fee_entry": fee_entry,
        "fee_exit": exit_fee,
        "net": net,
        "trend_strength": df['trend_strength'].iloc[i],
        "atr_percentile": df['atr_percentile'].iloc[i]
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

print("\n===== BACKTEST RESULTS (V9 - ENHANCED) =====")
print(f"Symbol: {SYMBOL}")
print(f"Timeframe: {INTERVAL}")
print(f"Data period: {df['time'].iloc[0].strftime('%Y-%m-%d')} to {df['time'].iloc[-1].strftime('%Y-%m-%d')}")
print(f"Bars analyzed: {len(df):,}")
print(f"\n--- PERFORMANCE ---")
print(f"Trades executed: {len(res_df)}")
print(f"Starting capital: ${INITIAL_CAPITAL:.2f}")
print(f"Ending capital:  ${capital:.2f}")
print(f"Total Net P/L:   ${total_net:.2f} ({(total_net/INITIAL_CAPITAL)*100:.1f}%)")
print(f"Average trade P/L: ${avg_pl:.2f}")
print(f"Median trade P/L:  ${median_pl:.2f}")
print(f"Win rate: {win_rate:.2f}%")
print(f"Wins: {legs_stats['win']}, Losses: {legs_stats['loss']}")
print(f"Max drawdown: {max_dd*100:.2f}%")

if len(res_df) > 0:
    avg_rr = res_df['rr_ratio'].mean()
    avg_sl_mult = res_df['atr_sl_mult'].mean()
    print(f"\n--- ADAPTIVE PARAMETERS ---")
    print(f"Average RR ratio used: {avg_rr:.2f}")
    print(f"Average SL multiplier: {avg_sl_mult:.2f}")
    
    print(f"\n--- TRADE ANALYSIS ---")
    tp_trades = res_df[res_df['exit_type'] == 'tp']
    sl_trades = res_df[res_df['exit_type'] == 'sl']
    print(f"TP hits: {len(tp_trades)} ({len(tp_trades)/len(res_df)*100:.1f}%)")
    print(f"SL hits: {len(sl_trades)} ({len(sl_trades)/len(res_df)*100:.1f}%)")
    
    if len(sl_trades) > 0:
        print(f"\nStop Loss Analysis:")
        print(f"  Avg trend strength on SL: {sl_trades['trend_strength'].mean():.2f}")
        print(f"  Avg ATR percentile on SL: {sl_trades['atr_percentile'].mean():.1f}")
        print(f"  Avg score on SL trades: {sl_trades['score'].mean():.2f}")

print(f"\n--- FILTER STATISTICS ---")
print(f"Total bars processed: {filter_stats['total_bars']:,}")
print(f"Filtered by NaN/insufficient data: {filter_stats['filter_nan']:,}")
print(f"Filtered by low volume: {filter_stats['filter_low_volume']:,}")
print(f"Filtered by sideways market: {filter_stats['filter_sideways']:,}")
print(f"Filtered by weak ATR: {filter_stats['filter_weak_atr']:,}")
print(f"Filtered by HTF conflict: {filter_stats['filter_htf_conflict']:,}")
print(f"Filtered by OBV/Volume misalignment: {filter_stats['filter_obv_volume_conflict']:,}")
print(f"Filtered by cooldown: {filter_stats['filter_cooldown']:,}")
print(f"Filtered by momentum conflict: {filter_stats['filter_momentum_conflict']:,}")
print(f"Filtered by weak score (<{MIN_SCORE_THRESHOLD}): {filter_stats['weak_score']:,}")

signal_opportunities = filter_stats['total_bars'] - filter_stats['filter_nan']
if signal_opportunities > 0:
    selectivity = (len(res_df) / signal_opportunities) * 100
    print(f"\nSelectivity: {selectivity:.2f}% (took {len(res_df)} trades from {signal_opportunities:,} valid bars)")

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Equity curve
ax1.plot(equity, linewidth=2, color='blue')
ax1.fill_between(range(len(equity)), equity, INITIAL_CAPITAL, alpha=0.3)
ax1.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', label='Starting Capital')
ax1.set_title(f'Equity Curve - {SYMBOL} {INTERVAL} (V9 Enhanced)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Trade Number')
ax1.set_ylabel('Capital (USD)')
ax1.legend()
ax1.grid(alpha=0.3)

# Drawdown
ax2.fill_between(range(len(drawdowns)), drawdowns*100, 0, color='red', alpha=0.5)
ax2.set_title('Drawdown %', fontsize=12)
ax2.set_xlabel('Trade Number')
ax2.set_ylabel('Drawdown %')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Save results
filename = f"backtest_{SYMBOL.replace('-', '_')}_{INTERVAL}_v9.csv"
res_df.to_csv(filename, index=False)
print(f"\nResults saved to: {filename}")

# Additional analysis: Win/Loss by score ranges
if len(res_df) > 0:
    print("\n--- SCORE ANALYSIS ---")
    score_ranges = [(9, 11), (11, 13), (13, 15), (15, 100)]
    for low, high in score_ranges:
        range_trades = res_df[(res_df['score'].abs() >= low) & (res_df['score'].abs() < high)]
        if len(range_trades) > 0:
            range_wr = (range_trades['net'] > 0).sum() / len(range_trades) * 100
            print(f"Score {low}-{high}: {len(range_trades)} trades, WR: {range_wr:.1f}%")
