"""
Backtest engine module for V4.
Single Responsibility: Execute backtest logic and generate performance metrics.

V4 CHANGES:
- Uses fixed stop loss based on support/resistance zones
- Targets zones (not lines) with RR-based stop calculation
- BULL: Target resistance zone, stop calculated from RR ratio
- BEAR: Target support zone, stop calculated from RR ratio
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from config import (
    INITIAL_CAPITAL, LEVERAGE, TAKER_FEE_RATE, SLIPPAGE_PCT,
    TARGET_RR_RATIO, MIN_RISK_PCT, MAX_TRADE_DURATION,
    MAX_MARGIN, RESULTS_DIR, INTERVAL, ZONE_APPROACH_PCT,
    SR_LOOKBACK, SR_MIN_TOUCHES, SR_ZONE_TOLERANCE, ZONE_TARGET_PCT
)
from data_loader import load_from_yfinance
from indicators import compute_indicators
from bias_scoring import compute_bias_vectorized
from trading import check_stops_vectorized
from support_resistance import calculate_sr_for_bar, calculate_trade_levels_v4, is_approaching_zone
from visualization import plot_equity_curve, plot_trade_distribution


def run_backtest(symbol, interval, limit, period):
    """
    Execute full backtest for a single symbol using V4 strategy.
    
    V4 Strategy:
    - Fixed stop loss based on RR ratio from target zone
    - BULL: Enter approaching resistance, target resistance zone
    - BEAR: Enter approaching support, target support zone
    
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
    
    # Calculate all bias values at once (vectorized)
    print(f"Computing bias scores for {len(df):,} bars...")
    bias_series = compute_bias_vectorized(df)
    df['bias'] = bias_series
    
    # Initialize S/R zone columns
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
    current_trade = None

    # Find all potential entry points upfront
    entry_indices = df[df['bias'].notna()].index.tolist()
    
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
            target_zone = current_trade['target_zone']
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
                    "target_zone": target_zone,
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
            
            # Continue to next bar
            continue
        
        # No open trade - check for entry signal
        if i not in entry_indices:
            continue
        
        if i >= len(df) - 1:
            continue
        
        bias = df['bias'].iloc[i]
        
        # Calculate S/R zones for THIS bar only, using ONLY historical data
        sr_zones = calculate_sr_for_bar(df, i, SR_LOOKBACK, SR_MIN_TOUCHES, SR_ZONE_TOLERANCE)
        
        if sr_zones is None:
            continue
        
        # Store S/R zones
        df.loc[df.index[i], 'support_zone_low'] = sr_zones['support_low']
        df.loc[df.index[i], 'support_zone_high'] = sr_zones['support_high']
        df.loc[df.index[i], 'resistance_zone_low'] = sr_zones['resistance_low']
        df.loc[df.index[i], 'resistance_zone_high'] = sr_zones['resistance_high']
        
        # Get current price (next bar open for realistic execution)
        current_price = float(df["close"].iloc[i])
        entry_price = float(df["open"].iloc[i+1])
        
        # Prepare zone dictionaries
        support_zone = {
            'low': sr_zones['support_low'],
            'high': sr_zones['support_high']
        } if sr_zones['support_low'] is not None else None
        
        resistance_zone = {
            'low': sr_zones['resistance_low'],
            'high': sr_zones['resistance_high']
        } if sr_zones['resistance_low'] is not None else None
        
        # Calculate trade levels using V4 logic
        trade_levels = calculate_trade_levels_v4(
            current_price=current_price,
            bias=bias,
            support_zone=support_zone,
            resistance_zone=resistance_zone,
            target_rr=TARGET_RR_RATIO,
            zone_approach_pct=ZONE_APPROACH_PCT,
            zone_target_pct=ZONE_TARGET_PCT,
            min_risk_pct=MIN_RISK_PCT
        )
        
        if trade_levels is None:
            # No valid trade setup
            continue
        
        # Extract trade parameters
        stop_price = trade_levels['stop']
        tp_price = trade_levels['tp']
        risk_amount = trade_levels['risk']
        reward_amount = trade_levels['reward']
        rr_ratio = trade_levels['rr_ratio']
        target_zone = trade_levels['target_zone']

        # Position sizing
        notional = capital if capital < MAX_MARGIN else MAX_MARGIN
        size = (notional * LEVERAGE) / entry_price
        
        if size <= 0:
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
            'fee_entry': fee_entry,
            'target_zone': target_zone,
            'rr_ratio': rr_ratio
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
