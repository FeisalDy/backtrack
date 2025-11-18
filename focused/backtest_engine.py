"""
Backtest engine module.
Single Responsibility: Execute backtest logic and generate performance metrics.
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from config import (
    INITIAL_CAPITAL, LEVERAGE, TAKER_FEE_RATE, SLIPPAGE_PCT,
    RR_FOLLOW, ALLOW_OVERLAP, MAX_BARS_AHEAD,
    MAX_MARGIN, RESULTS_DIR, INTERVAL, ZONE_PENETRATION,
    SR_LOOKBACK, SR_MIN_TOUCHES
)
from data_loader import load_from_yfinance
from indicators import compute_indicators
from bias_scoring import compute_bias_vectorized
from trading import check_stops_vectorized
from support_resistance import find_support_resistance_zones, get_entry_from_zone, validate_risk_reward
from visualization import plot_equity_curve, plot_trade_distribution


def run_backtest(symbol, interval, limit, period):
    """
    Execute full backtest for a single symbol.
    
    Optimizations applied:
    - Vectorized bias calculation
    - Vectorized stop checking
    - Memory-efficient data types
    - Pre-allocated arrays where possible
    
    Args:
        symbol: Trading pair symbol
        interval: Timeframe
        limit: Maximum number of bars
        
    Returns:
        Dictionary with backtest summary statistics
    """
    # Load and prepare data
    df = load_from_yfinance(symbol, interval, limit, period)
    df = compute_indicators(df)
    
    # Calculate all bias values at once (vectorized)
    print(f"Computing bias scores for {len(df):,} bars...")
    bias_series = compute_bias_vectorized(df)
    df['bias'] = bias_series
    
    # Find support and resistance zones
    print(f"Detecting support/resistance zones...")
    df = find_support_resistance_zones(df, lookback=SR_LOOKBACK, min_touches=SR_MIN_TOUCHES)

    # Initialize trading state
    capital = INITIAL_CAPITAL
    equity_curve = [capital]
    results = []
    legs_stats = {"win": 0, "loss": 0}
    trade_num = 0

    # Find all potential entry points upfront
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
        
        # Get entry and stop from S/R zone
        zone_levels = get_entry_from_zone(df, i, bias, ZONE_PENETRATION)
        
        if zone_levels is None:
            # No valid S/R zone found, skip this signal
            i_idx += 1
            continue
        
        entry_price = zone_levels['entry']
        stop_price = zone_levels['stop']
        
        # Validate risk-reward ratio
        rr_validation = validate_risk_reward(entry_price, stop_price, bias, RR_FOLLOW)
        
        if rr_validation is None:
            # RR ratio too low, skip this trade
            i_idx += 1
            continue
        
        tp_price = rr_validation['tp']
        risk_amount = rr_validation['risk']
        rr_ratio = rr_validation['rr_ratio']

        # Position sizing with margin limit
        notional = capital if capital < MAX_MARGIN else MAX_MARGIN
        size = (notional * LEVERAGE) / entry_price
        
        if size <= 0:
            i_idx += 1
            continue

        # Entry fee
        entry_notional = entry_price * size
        fee_entry = entry_notional * TAKER_FEE_RATE
        capital -= fee_entry

        # Check for TP/SL hits (vectorized)
        max_bars_ahead = min(MAX_BARS_AHEAD, len(df) - i - 2)
        df_slice = df.iloc[i+2:i+2+max_bars_ahead]
        
        if len(df_slice) == 0:
            i_idx += 1
            continue
        
        rel_idx, exit_type = check_stops_vectorized(df_slice, tp_price, stop_price, bias)
        
        if rel_idx >= 0:
            # Stop was hit
            abs_idx = i + 2 + rel_idx
            if exit_type == "tp":
                close_price = tp_price * ((1 + SLIPPAGE_PCT) if bias=="bull" else (1 - SLIPPAGE_PCT))
            else:  # sl
                close_price = stop_price * ((1 - SLIPPAGE_PCT) if bias=="bull" else (1 + SLIPPAGE_PCT))
        else:
            # No stop hit within max_bars_ahead - mark-to-market exit
            final_price = float(df["close"].iloc[min(i+2+max_bars_ahead-1, len(df)-1)])
            close_price = final_price * ((1 + SLIPPAGE_PCT) if bias=="bull" else (1 - SLIPPAGE_PCT))
            exit_type = "mtm"
            abs_idx = len(df) - 1

        # Calculate P&L
        pl = (close_price - entry_price) * size if bias=="bull" else (entry_price - close_price) * size
        exit_fee = abs(close_price*size)*TAKER_FEE_RATE
        capital += pl
        capital -= exit_fee

        # Update statistics
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
            "zone": zone_levels['zone'],
            "rr_ratio": rr_ratio,
            "size": size,
            "entry": entry_price,
            "stop_loss": stop_price,
            "take_profit": tp_price,
            "close_price": close_price,
            "exit_type": exit_type,
            "pl": pl,
            "fee_entry": fee_entry,
            "fee_exit": exit_fee,
            "net": pl - exit_fee,
            "balance": capital
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

    # Save results to CSV
    res_df = pd.DataFrame(results)
    csv_path = os.path.join(RESULTS_DIR, f"{interval}_{symbol}_backtest.csv")
    res_df.to_csv(csv_path, index=False)
    print(f"\nResults for {symbol} saved to: {csv_path}")

    # Generate visualizations
    print(f"Generating charts for {symbol}...")
    plot_equity_curve(equity_curve, symbol, INITIAL_CAPITAL)
    plot_trade_distribution(res_df, symbol)
    print(f"✅ Charts saved for {symbol}")

    # Calculate performance metrics
    net_pls: np.ndarray = res_df["net"].to_numpy(dtype=float) if not res_df.empty else np.array([], dtype=float)
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


def backtest_single_symbol(args):
    """
    Wrapper for parallel processing.
    
    Args:
        args: Tuple of (symbol, interval, limit)
        
    Returns:
        Backtest result dictionary or None on error
    """
    symbol, interval, limit, period = args
    try:
        return run_backtest(symbol, interval, limit, period)
    except Exception as e:
        print(f"Error backtesting {symbol}: {e}")
        return None
