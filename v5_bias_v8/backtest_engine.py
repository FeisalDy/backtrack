"""
Backtest engine module for V5 (bias_v8 logic).
Single Responsibility: Execute backtest logic and generate performance metrics.
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from config import (
    INITIAL_CAPITAL, LEVERAGE, TAKER_FEE_RATE, SLIPPAGE_PCT,
    RR_FOLLOW, ATR_SL_MULT, MAX_TRADE_DURATION,
    RESULTS_DIR, INTERVAL, ALLOW_OVERLAP
)
from data_loader import load_from_yfinance
from indicators import compute_indicators
from bias_scoring import compute_bias_vectorized
from trading import calculate_trade_levels, check_stops_vectorized
from visualization import plot_equity_curve, plot_trade_distribution


def run_backtest(symbol, interval, limit, period):
    """
    Execute full backtest for a single symbol using V5 (bias_v8) strategy.
    
    Strategy:
    - Enhanced bias scoring with 7 components
    - ATR-based stop loss
    - Fixed risk-reward ratio
    - No overlapping trades
    
    Args:
        symbol: Trading pair symbol
        interval: Timeframe
        limit: Maximum number of bars
        period: Data period to fetch
        
    Returns:
        Dictionary with backtest summary statistics
    """
    # Load and prepare data
    df = load_from_yfinance(symbol, interval, limit, period)
    df = compute_indicators(df)
    
    # Calculate all bias values at once
    print(f"Computing bias scores for {len(df):,} bars...")
    bias_series = compute_bias_vectorized(df)
    df['bias'] = bias_series

    # Initialize trading state
    capital = INITIAL_CAPITAL
    equity_curve = [capital]
    results = []
    legs_stats = {"win": 0, "loss": 0}
    trade_num = 0
    
    # Track current open trade (only ONE trade at a time)
    current_trade = None

    # Find all potential entry points upfront
    entry_indices = df[df['bias'].notna()].index.tolist()
    
    i_idx = 0
    pbar = tqdm(total=len(df), desc=f"Backtesting {symbol}", unit="bar", leave=False)
    
    # Iterate through ALL bars
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
            # Priority 2: Check time-based exit
            elif bars_open >= MAX_TRADE_DURATION:
                if unrealized_pl > 0:
                    exit_type = "time_profit"
                    close_price = current_price * ((1 + SLIPPAGE_PCT) if bias=="bull" else (1 - SLIPPAGE_PCT))
                    exit_triggered = True
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
            
            # Continue to next bar
            continue
        
        # No open trade - check for entry signal
        if i not in entry_indices:
            continue
        
        if i >= len(df) - 1:
            continue
        
        bias = df['bias'].iloc[i]
        
        # Entry at NEXT BAR OPEN (realistic market order execution)
        entry_price = float(df["open"].iloc[i+1])
        atr_val = float(df["atr"].iloc[i])
        
        # Calculate trade levels using ATR
        levels = calculate_trade_levels(entry_price, atr_val, bias, RR_FOLLOW, ATR_SL_MULT)
        tp_price = levels['tp']
        stop_price = levels['sl']

        # Position sizing: Always trade 3% of current capital
        risk_capital = capital * 0.01  # 3% of current capital
        notional = risk_capital
        size = (notional * LEVERAGE) / entry_price
        
        if size <= 0 or capital <= 0:
            continue
        
        # Entry fee
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
            'fee_entry': fee_entry
        }
    
    pbar.close()

    # Create symbol-specific folder
    symbol_dir = os.path.join(RESULTS_DIR, symbol)
    os.makedirs(symbol_dir, exist_ok=True)

    # Save results to CSV
    res_df = pd.DataFrame(results)
    csv_path = os.path.join(symbol_dir, f"{interval}_{symbol}_backtest.csv")
    res_df.to_csv(csv_path, index=False)
    print(f"\nResults for {symbol} saved to: {csv_path}")

    # Generate visualizations
    print(f"Generating charts for {symbol}...")
    plot_equity_curve(equity_curve, symbol, INITIAL_CAPITAL, symbol_dir)
    plot_trade_distribution(res_df, symbol, symbol_dir)
    print(f"✅ Charts saved for {symbol}")

    # Calculate performance metrics
    net_pls = res_df["net"].to_numpy(dtype=float) if not res_df.empty else np.array([], dtype=float)
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
        args: Tuple of (symbol, interval, limit, period)
        
    Returns:
        Backtest result dictionary or None on error
    """
    symbol, interval, limit, period = args
    try:
        return run_backtest(symbol, interval, limit, period)
    except Exception as e:
        print(f"Error backtesting {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None
