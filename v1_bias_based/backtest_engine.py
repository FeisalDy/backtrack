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
    RR_FOLLOW, MIN_RISK_PCT, MAX_BARS_AHEAD, MAX_TRADE_DURATION,
    MAX_MARGIN, RESULTS_DIR, INTERVAL, ZONE_PENETRATION,
    SR_LOOKBACK, SR_MIN_TOUCHES
)
from data_loader import load_from_yfinance
from indicators import compute_indicators
from bias_scoring import compute_bias_vectorized
from trading import check_stops_vectorized
from support_resistance import calculate_sr_for_bar, get_entry_from_zone, validate_risk_reward
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
    
    # Initialize S/R zone columns (will be calculated dynamically)
    df['support_zone_low'] = np.nan
    df['support_zone_high'] = np.nan
    df['resistance_zone_low'] = np.nan
    df['resistance_zone_high'] = np.nan

    # Initialize trading state
    capital = INITIAL_CAPITAL
    equity_curve = [capital]
    results = []
    legs_stats = {"win": 0, "loss": 0}
    trade_num = 0
    
    # Track current open trade (only ONE trade at a time)
    current_trade = None  # {'entry_idx': int, 'entry_price': float, 'size': float, ...}

    # Find all potential entry points upfront
    entry_indices = df[df['bias'].notna()].index.tolist()
    
    i_idx = 0
    pbar = tqdm(total=len(df), desc=f"Backtesting {symbol}", unit="bar", leave=False)
    
    # Iterate through ALL bars (not just entry points) to check open trades
    for i in range(len(df)):
        pbar.update(1)
        
        # Check if we have an open trade
        if current_trade is not None:
            entry_idx = current_trade['entry_idx']
            entry_price = current_trade['entry_price']
            size = current_trade['size']
            tp_price = current_trade['tp_price']
            stop_price = current_trade['stop_price']
            bias = current_trade['bias']
            fee_entry = current_trade['fee_entry']
            zone = current_trade['zone']
            rr_ratio = current_trade['rr_ratio']
            
            # Check how long trade has been open
            bars_open = i - entry_idx
            
            # Calculate current unrealized P&L
            current_price = float(df["close"].iloc[i])
            unrealized_pl = (current_price - entry_price) * size if bias == "bull" else (entry_price - current_price) * size
            
            # Check if TP or SL hit
            bar_high = df['high'].iloc[i]
            bar_low = df['low'].iloc[i]
            
            tp_hit = (bar_high >= tp_price if bias == "bull" else bar_low <= tp_price)
            sl_hit = (bar_low <= stop_price if bias == "bull" else bar_high >= stop_price)
            
            exit_triggered = False
            exit_type = None
            close_price = None
            
            # Priority 1: Check TP/SL
            if tp_hit and sl_hit:
                # Both hit same bar - conservative: assume SL first
                exit_type = "sl"
                close_price = stop_price * ((1 - SLIPPAGE_PCT) if bias=="bull" else (1 + SLIPPAGE_PCT))
                exit_triggered = True
            elif tp_hit:
                exit_type = "tp"
                close_price = tp_price * ((1 + SLIPPAGE_PCT) if bias=="bull" else (1 - SLIPPAGE_PCT))
                exit_triggered = True
            elif sl_hit:
                exit_type = "sl"
                close_price = stop_price * ((1 - SLIPPAGE_PCT) if bias=="bull" else (1 + SLIPPAGE_PCT))
                exit_triggered = True
            # Priority 2: Check time-based exit (after MAX_TRADE_DURATION)
            elif bars_open >= MAX_TRADE_DURATION:
                if unrealized_pl > 0:
                    # In profit - close it
                    exit_type = "time_profit"
                    close_price = current_price * ((1 + SLIPPAGE_PCT) if bias=="bull" else (1 - SLIPPAGE_PCT))
                    exit_triggered = True
                # If in loss, close it
                elif unrealized_pl < 0:
                    exit_type = "time_loss"
                    close_price = current_price * ((1 - SLIPPAGE_PCT) if bias=="bull" else (1 + SLIPPAGE_PCT))
                    exit_triggered = True
                
            
            # Execute exit if triggered
            if exit_triggered:
                pl = (close_price - entry_price) * size if bias=="bull" else (entry_price - close_price) * size
                exit_fee = abs(close_price * size) * TAKER_FEE_RATE
                
                # Update capital
                capital += pl
                capital -= fee_entry
                capital -= exit_fee
                
                # Record trade
                if pl > 0:
                    legs_stats["win"] += 1
                else:
                    legs_stats["loss"] += 1
                
                trade_num += 1
                results.append({
                    "trade": trade_num,
                    "entry_index": entry_idx,
                    "entry_time": df["time"].iloc[entry_idx],
                    "exit_index": i,
                    "exit_time": df["time"].iloc[i],
                    "duration_bars": bars_open,
                    "bias": bias,
                    "zone": zone,
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
                    "net": pl - fee_entry - exit_fee,
                    "balance": capital
                })
                
                equity_curve.append(capital)
                
                # Clear current trade
                current_trade = None
            
            # Continue to next bar (don't try to open new trade while one is open)
            continue
        
        # No open trade - check for entry signal
        if i not in entry_indices:
            continue
        
        if i >= len(df) - 1:
            continue
        
        bias = df['bias'].iloc[i]
        
        # Calculate S/R zones for THIS bar only, using ONLY historical data
        sr_zones = calculate_sr_for_bar(df, i, SR_LOOKBACK, SR_MIN_TOUCHES)
        
        if sr_zones is None:
            # Not enough historical data yet
            continue
        
        # Store S/R zones in dataframe (only for current bar)
        df.loc[df.index[i], 'support_zone_low'] = sr_zones['support_low']
        df.loc[df.index[i], 'support_zone_high'] = sr_zones['support_high']
        df.loc[df.index[i], 'resistance_zone_low'] = sr_zones['resistance_low']
        df.loc[df.index[i], 'resistance_zone_high'] = sr_zones['resistance_high']
        
        # Entry at NEXT BAR OPEN (realistic market order execution)
        entry_price = float(df["open"].iloc[i+1])
        
        # Get S/R zone for calculating stop loss
        zone_levels = get_entry_from_zone(df, i, bias, ZONE_PENETRATION)
        
        if zone_levels is None:
            # No valid S/R zone found, skip this signal
            continue
        
        # Use zone-based stop loss, but enter at market
        stop_price = zone_levels['stop']
        
        # Validate risk-reward ratio
        rr_validation = validate_risk_reward(entry_price, stop_price, bias, RR_FOLLOW, MIN_RISK_PCT)
        
        if rr_validation is None:
            # RR ratio too low, skip this trade
            continue
        
        tp_price = rr_validation['tp']
        risk_amount = rr_validation['risk']
        rr_ratio = rr_validation['rr_ratio']

        # Position sizing - simple, no overlapping trades
        notional = capital if capital < MAX_MARGIN else MAX_MARGIN
        size = (notional * LEVERAGE) / entry_price
        
        if size <= 0:
            continue
        
        # Entry fee (will be deducted when trade closes)
        entry_notional = entry_price * size
        fee_entry = entry_notional * TAKER_FEE_RATE
        
        # Open the trade
        current_trade = {
            'entry_idx': i + 1,
            'entry_price': entry_price,
            'size': size,
            'tp_price': tp_price,
            'stop_price': stop_price,
            'bias': bias,
            'fee_entry': fee_entry,
            'zone': zone_levels['zone'],
            'rr_ratio': rr_ratio
        }
    
    pbar.close()

    # Create symbol-specific folder
    symbol_dir = os.path.join(RESULTS_DIR, symbol)
    os.makedirs(symbol_dir, exist_ok=True)

    # Save results to CSV in symbol folder
    res_df = pd.DataFrame(results)
    csv_path = os.path.join(symbol_dir, f"{interval}_{symbol}_backtest.csv")
    res_df.to_csv(csv_path, index=False)
    print(f"\nResults for {symbol} saved to: {csv_path}")

    # Generate visualizations in symbol folder
    print(f"Generating charts for {symbol}...")
    plot_equity_curve(equity_curve, symbol, INITIAL_CAPITAL, symbol_dir)
    plot_trade_distribution(res_df, symbol, symbol_dir)
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
