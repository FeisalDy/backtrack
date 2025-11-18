# ===== BACKTEST RESULTS =====
# Symbol: BTC-USD/USD
# Timeframe: 30m
# Data period: 2025-09-18 to 2025-11-17
# Bars analyzed: 2,858
# Trades executed: 43
# Starting capital: $20.00
# Ending capital:  $30.86
# Total Net P/L:   $10.86
# Average trade P/L: $0.27
# Median trade P/L:  $0.18
# Win rate: 51.16%
# Legs: wins=22, losses=21
# Max drawdown: 8.90%

# ===== CRYPTO TRADING BOT BACKTEST (YFinance Version) =====
# Fetch data from Yahoo Finance instead of parquet files

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

RR_FOLLOW = 2.5
ATR_SL_MULT = 2.0

ALLOW_OVERLAP = True

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
    df["atr_median"] = df["atr"].rolling(50).median()
    
    # Price Momentum
    df['price_momentum_3'] = (df['close'] - df['close'].shift(3)) / df['close'].shift(3) * 100
    df['price_momentum_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100
    
    # Recent High/Low Breakout
    df['recent_high_5'] = df['high'].rolling(5).max()
    df['recent_low_5'] = df['low'].rolling(5).min()
    df['recent_high_20'] = df['high'].rolling(20).max()
    df['recent_low_20'] = df['low'].rolling(20).min()
    
    # Volume Surge
    volume_avg_10 = df['volume'].rolling(10).mean()
    df['volume_surge'] = df['volume'] / volume_avg_10
    
    # Price vs Recent Range
    df['price_in_range_5'] = (df['close'] - df['recent_low_5']) / (df['recent_high_5'] - df['recent_low_5'])
    df['price_in_range_20'] = (df['close'] - df['recent_low_20']) / (df['recent_high_20'] - df['recent_low_20'])
    
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
    
    return df

df = compute_indicators(df)

# -------------------------
# BIAS FUNCTION
# -------------------------
def get_bias(df, i):
    if i < 100:
        return None

    # Check for NaN values
    required_cols = ['price_in_range_5', 'price_in_range_20', 'recent_high_5', 'recent_low_5',
                    'recent_high_20', 'recent_low_20', 'volume_surge', 'atr', 'atr_median',
                    'price_momentum_3_zscore', 'price_momentum_10_zscore', 'volume_surge_zscore',
                    'obv_momentum_10_zscore', 'obv_momentum_14_zscore', 'trend_strength',
                    'fast_ma', 'slow_ma', 'medium_ma']
    
    for col in required_cols:
        if pd.isna(df[col].iloc[i]):
            return None

    # Only trade when ATR > median (volatile enough)
    if df['atr'].iloc[i] <= df['atr_median'].iloc[i]:
        return None

    score = 0.0
    
    # 1. Price position in range
    price_range_5 = df['price_in_range_5'].iloc[i]
    if price_range_5 > 0.75:
        score += 2.5
    elif price_range_5 < 0.25:
        score -= 2.5
    elif price_range_5 > 0.6:
        score += 1.5
    elif price_range_5 < 0.4:
        score -= 1.5

    # 2. Multi-timeframe breakout
    short_breakout_bull = df['close'].iloc[i] > df['recent_high_5'].iloc[i-1]
    short_breakout_bear = df['close'].iloc[i] < df['recent_low_5'].iloc[i-1]
    medium_trend_bull = df['close'].iloc[i] > df['recent_high_20'].iloc[i-1] * 0.995
    medium_trend_bear = df['close'].iloc[i] < df['recent_low_20'].iloc[i-1] * 1.005
    
    if short_breakout_bull and medium_trend_bull:
        score += 4.0
    elif short_breakout_bear and medium_trend_bear:
        score -= 4.0
    elif short_breakout_bull:
        score += 1.5
    elif short_breakout_bear:
        score -= 1.5

    # 3. Price momentum
    momentum_3_z = df['price_momentum_3_zscore'].iloc[i]
    momentum_10_z = df['price_momentum_10_zscore'].iloc[i]
    
    if momentum_3_z > 1.0 and momentum_10_z > 0.5:
        score += 3.0
    elif momentum_3_z < -1.0 and momentum_10_z < -0.5:
        score -= 3.0
    elif momentum_3_z > 0.5:
        score += 1.5
    elif momentum_3_z < -0.5:
        score -= 1.5

    # 4. Volume confirmation
    volume_z = df['volume_surge_zscore'].iloc[i]
    price_up = df['price_momentum_3_zscore'].iloc[i] > 0.5
    price_down = df['price_momentum_3_zscore'].iloc[i] < -0.5
    
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

    # 5. OBV momentum
    obv_10_z = df['obv_momentum_10_zscore'].iloc[i]
    obv_14_z = df['obv_momentum_14_zscore'].iloc[i]
    
    if obv_10_z > 0.5 and obv_14_z > 0.5:
        score += 2.0
    elif obv_10_z < -0.5 and obv_14_z < -0.5:
        score -= 2.0
    elif obv_10_z > 1.0 or obv_14_z > 1.0:
        score += 1.0
    elif obv_10_z < -1.0 or obv_14_z < -1.0:
        score -= 1.0

    # 6. Trend strength
    trend_str = df['trend_strength'].iloc[i]
    if abs(trend_str) > 2.0:
        if trend_str > 0:
            score += 2.0
        else:
            score -= 2.0
    elif abs(trend_str) > 1.0:
        if trend_str > 0:
            score += 1.0
        else:
            score -= 1.0

    # 7. MA confirmation
    if df['fast_ma'].iloc[i] > df['slow_ma'].iloc[i] > df['medium_ma'].iloc[i]:
        score += 1.5
    elif df['fast_ma'].iloc[i] < df['slow_ma'].iloc[i] < df['medium_ma'].iloc[i]:
        score -= 1.5

    # Bias determination
    if score >= 8.0:
        return "bull"
    elif score <= -8.0:
        return "bear"
    else:
        return None

# -------------------------
# TRADE LEVELS
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

print("\n===== BACKTEST RESULTS =====")
print(f"Symbol: {SYMBOL}/USD")
print(f"Timeframe: {INTERVAL}")
print(f"Data period: {df['time'].iloc[0].strftime('%Y-%m-%d')} to {df['time'].iloc[-1].strftime('%Y-%m-%d')}")
print(f"Bars analyzed: {len(df):,}")
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
plt.title(f'Equity Curve - {SYMBOL}/USD {INTERVAL}')
plt.xlabel('Trade number')
plt.ylabel('Capital (USD)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

res_df.to_csv(f"backtest_{SYMBOL}_{INTERVAL}.csv", index=False)
print(f"\nResults saved to: backtest_{SYMBOL}_{INTERVAL}.csv")
