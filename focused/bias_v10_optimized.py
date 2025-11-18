import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yfinance as yf
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, "results")
os.makedirs(results_dir, exist_ok=True)

# -------------------------
# USER PARAMETERS
# -------------------------
SYMBOLS = ["BTC-USD", "ETH-USD", "DOGE-USD", "SOL-USD", "ADA-USD", "XRP-USD", "BNB-USD", "TRX-USD", "LINK-USD"]
INTERVAL = "1h"      # "5m", "15m", "30m", "1h"
LIMIT = 100000       # Max bars to use for backtest

INITIAL_CAPITAL = 20.0
LEVERAGE = 2

TAKER_FEE_RATE = 0.0004
SLIPPAGE_PCT = 0.0002

RR_FOLLOW = 2.5
ATR_SL_MULT = 2.0

ALLOW_OVERLAP = True
MAX_WORKERS = min(4, mp.cpu_count())  # Limit concurrent downloads

# -------------------------
# DATA LOADING
# -------------------------
def load_from_yfinance(symbol, interval, limit):
    print(f"Fetching {symbol} {interval} data from Yahoo Finance...")
    df = yf.download(symbol, period="max", interval=interval, progress=False)
    
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
    
    # ✅ OPTIMIZATION: Downcast to float32 to save ~50% memory
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype('float32')
    df['volume'] = df['volume'].astype('int64')
    
    print(f"Loaded {len(df):,} bars | {df['time'].iloc[0].strftime('%Y-%m-%d')} to {df['time'].iloc[-1].strftime('%Y-%m-%d')}")
    return df

# -------------------------
# INDICATORS (VECTORIZED)
# -------------------------
def compute_indicators(df):
    """✅ OPTIMIZATION: All calculations are vectorized"""
    
    # Moving Averages
    df["fast_ma"] = df["close"].rolling(20).mean()
    df["slow_ma"] = df["close"].rolling(50).mean()
    df["medium_ma"] = df["close"].rolling(100).mean()
    
    # ✅ CRITICAL FIX: Vectorized OBV calculation (replaces loop)
    # Original loop iterated 100k times - now instant!
    delta_close = df["close"].diff()
    volume_signed = df["volume"].copy()
    volume_signed[delta_close < 0] = -volume_signed[delta_close < 0]
    volume_signed[delta_close == 0] = 0
    df["obv"] = volume_signed.cumsum()
    
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
    
    # OBV Momentum (now instant with vectorized OBV)
    df['obv_momentum_10'] = df['obv'].diff(10)
    df['obv_momentum_14'] = df['obv'].diff(14)
    
    # Z-score normalization
    for col in ['price_momentum_3', 'price_momentum_10', 'volume_surge', 'obv_momentum_10', 'obv_momentum_14']:
        rolling_mean = df[col].rolling(50).mean()
        rolling_std = df[col].rolling(50).std()
        df[f'{col}_zscore'] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
    
    # Trend strength
    df['trend_strength'] = (df['fast_ma'] - df['slow_ma']) / (df['atr'] + 1e-8)
    
    # ✅ OPTIMIZATION: Downcast indicator columns to save memory
    for col in df.columns:
        if col not in ['time', 'volume'] and df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
    
    return df

# -------------------------
# VECTORIZED BIAS SCORING
# -------------------------
def compute_bias_vectorized(df):
    """
    ✅ MAJOR OPTIMIZATION: Vectorized bias calculation
    - Original: Called get_bias() for each bar individually
    - Optimized: Calculate all bias scores at once using NumPy
    - Speed improvement: ~50-100x faster
    """
    n = len(df)
    bias = pd.Series([None] * n, index=df.index)
    
    # Pre-filter: Only calculate bias where ATR > median and index >= 100
    valid_mask = (df.index >= 100) & (df['atr'] > df['atr_median'])
    
    # Check for required columns (vectorized)
    required_cols = ['price_in_range_5', 'price_in_range_20', 'recent_high_5', 'recent_low_5',
                    'recent_high_20', 'recent_low_20', 'volume_surge', 'atr', 'atr_median',
                    'price_momentum_3_zscore', 'price_momentum_10_zscore', 'volume_surge_zscore',
                    'obv_momentum_10_zscore', 'obv_momentum_14_zscore', 'trend_strength',
                    'fast_ma', 'slow_ma', 'medium_ma']
    
    # Remove rows with NaN in required columns
    for col in required_cols:
        valid_mask = valid_mask & df[col].notna()
    
    if valid_mask.sum() == 0:
        return bias
    
    # Initialize score array
    score = np.zeros(n, dtype=np.float32)
    
    # Extract columns for vectorized operations (only valid rows)
    idx = valid_mask.values
    price_range_5 = df.loc[idx, 'price_in_range_5'].values
    close = df.loc[idx, 'close'].values
    recent_high_5_prev = df.loc[idx, 'recent_high_5'].shift(1).values
    recent_low_5_prev = df.loc[idx, 'recent_low_5'].shift(1).values
    recent_high_20_prev = df.loc[idx, 'recent_high_20'].shift(1).values
    recent_low_20_prev = df.loc[idx, 'recent_low_20'].shift(1).values
    momentum_3_z = df.loc[idx, 'price_momentum_3_zscore'].values
    momentum_10_z = df.loc[idx, 'price_momentum_10_zscore'].values
    volume_z = df.loc[idx, 'volume_surge_zscore'].values
    obv_10_z = df.loc[idx, 'obv_momentum_10_zscore'].values
    obv_14_z = df.loc[idx, 'obv_momentum_14_zscore'].values
    trend_str = df.loc[idx, 'trend_strength'].values
    fast_ma = df.loc[idx, 'fast_ma'].values
    slow_ma = df.loc[idx, 'slow_ma'].values
    medium_ma = df.loc[idx, 'medium_ma'].values
    
    # Vectorized scoring logic
    score_temp = np.zeros(idx.sum(), dtype=np.float32)
    
    # 1. Price position in range
    score_temp += np.where(price_range_5 > 0.75, 2.5, 0)
    score_temp += np.where(price_range_5 < 0.25, -2.5, 0)
    score_temp += np.where((price_range_5 > 0.6) & (price_range_5 <= 0.75), 1.5, 0)
    score_temp += np.where((price_range_5 < 0.4) & (price_range_5 >= 0.25), -1.5, 0)
    
    # 2. Multi-timeframe breakout
    short_breakout_bull = close > recent_high_5_prev
    short_breakout_bear = close < recent_low_5_prev
    medium_trend_bull = close > recent_high_20_prev * 0.995
    medium_trend_bear = close < recent_low_20_prev * 1.005
    
    score_temp += np.where(short_breakout_bull & medium_trend_bull, 4.0, 0)
    score_temp += np.where(short_breakout_bear & medium_trend_bear, -4.0, 0)
    score_temp += np.where(short_breakout_bull & ~medium_trend_bull, 1.5, 0)
    score_temp += np.where(short_breakout_bear & ~medium_trend_bear, -1.5, 0)
    
    # 3. Price momentum
    score_temp += np.where((momentum_3_z > 1.0) & (momentum_10_z > 0.5), 3.0, 0)
    score_temp += np.where((momentum_3_z < -1.0) & (momentum_10_z < -0.5), -3.0, 0)
    score_temp += np.where((momentum_3_z > 0.5) & ~((momentum_3_z > 1.0) & (momentum_10_z > 0.5)), 1.5, 0)
    score_temp += np.where((momentum_3_z < -0.5) & ~((momentum_3_z < -1.0) & (momentum_10_z < -0.5)), -1.5, 0)
    
    # 4. Volume confirmation
    price_up = momentum_3_z > 0.5
    price_down = momentum_3_z < -0.5
    score_temp += np.where((volume_z > 1.5) & price_up, 2.5, 0)
    score_temp += np.where((volume_z > 1.5) & price_down, -2.5, 0)
    score_temp += np.where((volume_z > 1.0) & (volume_z <= 1.5) & price_up, 1.5, 0)
    score_temp += np.where((volume_z > 1.0) & (volume_z <= 1.5) & price_down, -1.5, 0)
    
    # 5. OBV momentum
    score_temp += np.where((obv_10_z > 0.5) & (obv_14_z > 0.5), 2.0, 0)
    score_temp += np.where((obv_10_z < -0.5) & (obv_14_z < -0.5), -2.0, 0)
    score_temp += np.where(((obv_10_z > 1.0) | (obv_14_z > 1.0)) & ~((obv_10_z > 0.5) & (obv_14_z > 0.5)), 1.0, 0)
    score_temp += np.where(((obv_10_z < -1.0) | (obv_14_z < -1.0)) & ~((obv_10_z < -0.5) & (obv_14_z < -0.5)), -1.0, 0)
    
    # 6. Trend strength
    score_temp += np.where(trend_str > 2.0, 2.0, 0)
    score_temp += np.where(trend_str < -2.0, -2.0, 0)
    score_temp += np.where((trend_str > 1.0) & (trend_str <= 2.0), 1.0, 0)
    score_temp += np.where((trend_str < -1.0) & (trend_str >= -2.0), -1.0, 0)
    
    # 7. MA confirmation
    score_temp += np.where((fast_ma > slow_ma) & (slow_ma > medium_ma), 1.5, 0)
    score_temp += np.where((fast_ma < slow_ma) & (slow_ma < medium_ma), -1.5, 0)
    
    # Map scores to bias
    score[idx] = score_temp
    bias_values = np.where(score >= 8.0, 1, np.where(score <= -8.0, -1, 0))  # 1=bull, -1=bear, 0=None
    
    # Convert to string bias
    bias_map = {1: "bull", -1: "bear", 0: None}
    bias = pd.Series([bias_map[v] for v in bias_values], index=df.index)
    
    return bias

# -------------------------
# TRADE LEVELS
# -------------------------
def calculate_trade_levels(entry_price, atr_val, bias, rr_follow=2.0, atr_sl_mult=1.5):
    """No changes - already efficient"""
    if bias == "bull":
        stop = entry_price - atr_val * atr_sl_mult
        risk = entry_price - stop
        tp = entry_price + risk * rr_follow
    else:
        stop = entry_price + atr_val * atr_sl_mult
        risk = stop - entry_price
        tp = entry_price - risk * rr_follow
    return {"tp": tp, "sl": stop}

def check_stops_vectorized(df_slice, tp_level, sl_level, bias):
    """
    ✅ OPTIMIZATION: Check all bars at once instead of loop
    Returns index of first TP/SL hit, or -1 if none
    """
    high = df_slice['high'].values
    low = df_slice['low'].values
    
    if bias == "bull":
        tp_hit = high >= tp_level
        sl_hit = low <= sl_level
    else:
        tp_hit = low <= tp_level
        sl_hit = high >= sl_level
    
    # Find first occurrence
    tp_indices = np.where(tp_hit)[0]
    sl_indices = np.where(sl_hit)[0]
    
    if len(tp_indices) == 0 and len(sl_indices) == 0:
        return -1, None  # No hit
    elif len(tp_indices) == 0:
        return sl_indices[0], "sl"
    elif len(sl_indices) == 0:
        return tp_indices[0], "tp"
    else:
        # Both hit - check which came first
        if tp_indices[0] < sl_indices[0]:
            return tp_indices[0], "tp"
        elif sl_indices[0] < tp_indices[0]:
            return sl_indices[0], "sl"
        else:
            # Same bar - conservative: assume SL
            return sl_indices[0], "sl"

# -------------------------
# VISUALIZATION FUNCTIONS
# -------------------------
def plot_equity_curve(equity_curve, symbol, initial_capital):
    """Plot equity curve with drawdown"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    equity = np.array(equity_curve)
    trades = np.arange(len(equity))
    
    # Equity curve
    ax1.plot(trades, equity, linewidth=2, color='#2E86AB', label='Equity')
    ax1.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    ax1.fill_between(trades, initial_capital, equity, where=(equity >= initial_capital), 
                      alpha=0.3, color='green', label='Profit')
    ax1.fill_between(trades, initial_capital, equity, where=(equity < initial_capital), 
                      alpha=0.3, color='red', label='Loss')
    ax1.set_ylabel('Equity ($)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{symbol} - Equity Curve', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Drawdown
    running_max = np.maximum.accumulate(equity)
    drawdown = (running_max - equity) / running_max * 100
    ax2.fill_between(trades, 0, drawdown, color='red', alpha=0.3)
    ax2.plot(trades, drawdown, color='darkred', linewidth=1.5)
    ax2.set_xlabel('Trade Number', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'equity_curve_{symbol}_{INTERVAL}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_trade_distribution(res_df, symbol):
    """Plot P&L distribution and win/loss analysis"""
    if res_df.empty:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # P&L Distribution
    ax1 = axes[0, 0]
    net_pls = res_df['net'].values
    ax1.hist(net_pls, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
    ax1.axvline(x=np.mean(net_pls), color='green', linestyle='--', linewidth=2, label=f'Mean: ${np.mean(net_pls):.2f}')
    ax1.set_xlabel('Net P&L ($)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Trade P&L Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Cumulative P&L
    ax2 = axes[0, 1]
    cumulative_pl = np.cumsum(net_pls)
    ax2.plot(cumulative_pl, linewidth=2, color='#A23B72')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.fill_between(range(len(cumulative_pl)), 0, cumulative_pl, 
                      where=(cumulative_pl >= 0), alpha=0.3, color='green')
    ax2.fill_between(range(len(cumulative_pl)), 0, cumulative_pl, 
                      where=(cumulative_pl < 0), alpha=0.3, color='red')
    ax2.set_xlabel('Trade Number', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Cumulative P&L ($)', fontsize=11, fontweight='bold')
    ax2.set_title('Cumulative P&L', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Win/Loss by Bias
    ax3 = axes[1, 0]
    bias_results = res_df.groupby('bias').apply(
        lambda x: pd.Series({
            'wins': (x['net'] > 0).sum(),
            'losses': (x['net'] <= 0).sum()
        })
    )
    bias_results.plot(kind='bar', ax=ax3, color=['green', 'red'], alpha=0.7)
    ax3.set_xlabel('Bias', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax3.set_title('Wins vs Losses by Bias', fontsize=12, fontweight='bold')
    ax3.legend(['Wins', 'Losses'])
    ax3.grid(True, alpha=0.3, axis='y')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0)
    
    # Exit Type Distribution
    ax4 = axes[1, 1]
    exit_counts = res_df['exit_type'].value_counts()
    colors_exit = {'tp': 'green', 'sl': 'red', 'mtm': 'orange'}
    colors = [colors_exit.get(x, 'gray') for x in exit_counts.index]
    ax4.pie(exit_counts.values, labels=exit_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    ax4.set_title('Exit Type Distribution', fontsize=12, fontweight='bold')
    
    plt.suptitle(f'{symbol} - Trade Analysis', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'trade_analysis_{symbol}_{INTERVAL}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_performance_comparison(all_results):
    """Compare performance across all symbols"""
    if not all_results:
        return
    
    df = pd.DataFrame(all_results)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Net P&L comparison
    ax1 = axes[0, 0]
    net_pls = [float(x.replace('$', '')) for x in df['Net P/L']]
    colors = ['green' if x > 0 else 'red' for x in net_pls]
    ax1.barh(df['Symbol'], net_pls, color=colors, alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax1.set_xlabel('Net P&L ($)', fontsize=11, fontweight='bold')
    ax1.set_title('Net P&L by Symbol', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Win Rate comparison
    ax2 = axes[0, 1]
    win_rates = [float(x) for x in df['Win Rate %']]
    ax2.barh(df['Symbol'], win_rates, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(x=50, color='orange', linestyle='--', linewidth=2, label='50% Threshold')
    ax2.set_xlabel('Win Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Win Rate by Symbol', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Max Drawdown comparison
    ax3 = axes[1, 0]
    max_dds = [float(x) for x in df['Max DD %']]
    ax3.barh(df['Symbol'], max_dds, color='darkred', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Max Drawdown (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Maximum Drawdown by Symbol', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Number of Trades
    ax4 = axes[1, 1]
    ax4.barh(df['Symbol'], df['Trades'], color='purple', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Number of Trades', fontsize=11, fontweight='bold')
    ax4.set_title('Trade Count by Symbol', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle(f'Performance Comparison - {INTERVAL} Timeframe', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'performance_comparison_{INTERVAL}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Performance comparison chart saved: {os.path.join(results_dir, f'performance_comparison_{INTERVAL}.png')}")

# -------------------------
# BACKTEST (OPTIMIZED)
# -------------------------
def run_backtest(symbol, interval, limit):
    """
    ✅ OPTIMIZATIONS APPLIED:
    1. Vectorized bias calculation
    2. Vectorized stop checking
    3. Memory-efficient data types
    4. Pre-allocated arrays where possible
    """
    df = load_from_yfinance(symbol, interval, limit)
    df = compute_indicators(df)
    
    # ✅ Calculate all bias values at once
    print(f"Computing bias scores for {len(df):,} bars...")
    bias_series = compute_bias_vectorized(df)
    df['bias'] = bias_series

    capital = INITIAL_CAPITAL
    equity_curve = [capital]
    results = []
    legs_stats = {"win":0, "loss":0}
    trade_num = 0

    # ✅ Find all potential entry points upfront
    entry_indices = df[df['bias'].notna()].index.tolist()
    
    i_idx = 0
    pbar = tqdm(total=len(entry_indices), desc=f"Backtesting {symbol}", unit="trade", leave=False)
    
    while i_idx < len(entry_indices):
        i = entry_indices[i_idx]
        pbar.update(1)
        
        if i >= len(df) - 1:
            i_idx += 1
            continue
        
        bias = df['bias'].iloc[i]
        
        entry_price = float(df["open"].iloc[i+1])
        atr_val = float(df["atr"].iloc[i])
        levels = calculate_trade_levels(entry_price, atr_val, bias, RR_FOLLOW, ATR_SL_MULT)

        MAX_MARGIN = 100.0  # USD
        notional = capital if capital < MAX_MARGIN else MAX_MARGIN
        size = (notional * LEVERAGE) / entry_price
        
        if size <= 0:
            i_idx += 1
            continue

        entry_notional = entry_price * size
        fee_entry = entry_notional * TAKER_FEE_RATE
        capital -= fee_entry

        # ✅ OPTIMIZATION: Vectorized stop checking
        # Check next 500 bars max (configurable limit to prevent checking entire dataset)
        max_bars_ahead = min(500, len(df) - i - 2)
        df_slice = df.iloc[i+2:i+2+max_bars_ahead]
        
        if len(df_slice) == 0:
            i_idx += 1
            continue
        
        rel_idx, exit_type = check_stops_vectorized(df_slice, levels["tp"], levels["sl"], bias)
        
        if rel_idx >= 0:
            # Stop was hit
            abs_idx = i + 2 + rel_idx
            if exit_type == "tp":
                close_price = levels["tp"] * ((1 + SLIPPAGE_PCT) if bias=="bull" else (1 - SLIPPAGE_PCT))
            else:  # sl
                close_price = levels["sl"] * ((1 - SLIPPAGE_PCT) if bias=="bull" else (1 + SLIPPAGE_PCT))
        else:
            # No stop hit within max_bars_ahead - mark-to-market
            final_price = float(df["close"].iloc[min(i+2+max_bars_ahead-1, len(df)-1)])
            close_price = final_price * ((1 + SLIPPAGE_PCT) if bias=="bull" else (1 - SLIPPAGE_PCT))
            exit_type = "mtm"
            abs_idx = len(df) - 1

        pl = (close_price - entry_price) * size if bias=="bull" else (entry_price - close_price) * size
        exit_fee = abs(close_price*size)*TAKER_FEE_RATE
        capital += pl
        capital -= exit_fee

        if pl > 0:
            legs_stats["win"] += 1
        else:
            legs_stats["loss"] += 1

        trade_num += 1
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
        
        # Move to next entry
        if ALLOW_OVERLAP:
            i_idx += 1
        else:
            # Skip to after this trade closes
            next_i = abs_idx
            while i_idx < len(entry_indices) and entry_indices[i_idx] <= next_i:
                i_idx += 1

    pbar.close()

    res_df = pd.DataFrame(results)
    csv_path = os.path.join(results_dir, f"backtest_{symbol}_{interval}.csv")
    res_df.to_csv(csv_path, index=False)
    print(f"\nResults for {symbol} saved to: {csv_path}")

    # ✅ Generate visualizations
    print(f"Generating charts for {symbol}...")
    plot_equity_curve(equity_curve, symbol, INITIAL_CAPITAL)
    plot_trade_distribution(res_df, symbol)
    print(f"✅ Charts saved for {symbol}")

    net_pls = res_df["net"].values if not res_df.empty else np.array([])
    total_net = capital - INITIAL_CAPITAL
    avg_pl = np.mean(net_pls) if len(net_pls)>0 else 0.0
    median_pl = np.median(net_pls) if len(net_pls)>0 else 0.0
    win_rate = np.mean([1 if x>0 else 0 for x in net_pls])*100 if len(net_pls)>0 else 0.0

    equity = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity)
    drawdowns = (running_max - equity)/running_max
    max_dd = np.max(drawdowns) if len(drawdowns)>0 else 0.0

    return {
        "Symbol": symbol,
        "Timeframe": interval,
        "Start Date": df['time'].iloc[0].strftime('%Y-%m-%d'),
        "End Date": df['time'].iloc[-1].strftime('%Y-%m-%d'),
        "Bars": len(df),
        "Trades": len(res_df),
        "Start Cap": f"${INITIAL_CAPITAL:.2f}",
        "End Cap": f"${capital:.2f}",
        "Net P/L": f"${total_net:.2f}",
        "Avg P/L": f"${avg_pl:.2f}",
        "Win Rate %": f"{win_rate:.2f}",
        "Wins": legs_stats['win'],
        "Losses": legs_stats['loss'],
        "Max DD %": f"{max_dd*100:.2f}"
    }

# -------------------------
# PARALLEL EXECUTION
# -------------------------
def backtest_single_symbol(args):
    """Wrapper for parallel processing"""
    symbol, interval, limit = args
    try:
        return run_backtest(symbol, interval, limit)
    except Exception as e:
        print(f"Error backtesting {symbol}: {e}")
        return None

# -------------------------
# MAIN EXECUTION
# -------------------------
if __name__ == "__main__":
    print(f"Starting backtest with {MAX_WORKERS} parallel workers...")
    print(f"Results will be saved to: ./results/")
    
    # ✅ OPTIMIZATION: Parallel processing of symbols
    args_list = [(symbol, INTERVAL, LIMIT) for symbol in SYMBOLS]
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        all_results = list(executor.map(backtest_single_symbol, args_list))
    
    # Filter out None results (errors)
    all_results = [r for r in all_results if r is not None]

    if all_results:
        summary_df = pd.DataFrame(all_results)
        print("\n\n===== SUMMARY BACKTEST RESULTS =====")
        print(summary_df.to_string())
        
        # ✅ Generate comparison chart
        print("\n\nGenerating performance comparison chart...")
        plot_performance_comparison(all_results)