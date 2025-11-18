"""
Backtest engine V2 - Trailing Stop Strategy.
Single Responsibility: Execute backtest with pullback entries, swing-based stops, and trailing mechanism.
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from config import (
    INITIAL_CAPITAL, LEVERAGE, TAKER_FEE_RATE, SLIPPAGE_PCT,
    RR_FOLLOW, MIN_RISK_PCT, MAX_BARS_AHEAD,
    MAX_MARGIN, RESULTS_DIR, INTERVAL,
    SWING_LOOKBACK, SWING_TOLERANCE_PCT,
    PULLBACK_TO_ZONE_PCT, ZONE_DEPTH_PCT,
    TRAIL_ACTIVATION_R, TRAIL_DISTANCE_PCT, BREAKEVEN_AT_R,
    SR_LOOKBACK, SR_MIN_TOUCHES
)
from data_loader import load_from_yfinance
from indicators import compute_indicators
from bias_scoring import compute_bias_vectorized
from support_resistance import calculate_sr_for_bar
from swing_detection import (
    find_recent_swing_low, find_recent_swing_high,
    calculate_swing_stop, check_pullback_entry
)
from trailing_stop import update_trailing_stop, check_trailing_stop_hit
from visualization import plot_equity_curve, plot_trade_distribution


def run_backtest(symbol, interval, limit, period):
    """
    Execute V2 backtest with trailing stops.
    
    Key Changes from V1:
    - Pullback entry: Wait for price to enter S/R zone before entering
    - Swing-based stops: Use recent swing highs/lows with tolerance zones
    - Trailing stops: Move stop to breakeven then trail behind price
    - No time-based exits: Let price action decide when to exit
    
    Args:
        symbol: Trading pair symbol
        interval: Timeframe
        limit: Maximum number of bars
        period: Data period for yfinance
        
    Returns:
        Dictionary with backtest summary statistics
    """
    # Load and prepare data
    df = load_from_yfinance(symbol, interval, limit, period)
    df = compute_indicators(df)
    
    # Calculate all bias values
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
    
    # Find all potential entry signals
    entry_signals = df[df['bias'].notna()].index.tolist()
    
    pbar = tqdm(total=len(df), desc=f"Backtesting {symbol}", unit="bar", leave=False)
    
    # Iterate through all bars
    for i in range(len(df)):
        pbar.update(1)
        
        # ===============================================================
        # PART 1: MANAGE OPEN TRADE
        # ===============================================================
        if current_trade is not None:
            entry_idx = current_trade['entry_idx']
            entry_price = current_trade['entry_price']
            size = current_trade['size']
            initial_stop = current_trade['initial_stop']
            current_stop = current_trade['current_stop']
            tp_price = current_trade['tp_price']
            bias = current_trade['bias']
            fee_entry = current_trade['fee_entry']
            zone = current_trade['zone']
            rr_ratio = current_trade['rr_ratio']
            highest_since_entry = current_trade['highest_since_entry']
            lowest_since_entry = current_trade['lowest_since_entry']
            
            # Update highest/lowest since entry
            bar_high = df['high'].iloc[i]
            bar_low = df['low'].iloc[i]
            current_price = float(df['close'].iloc[i])
            
            if bar_high > highest_since_entry:
                highest_since_entry = bar_high
            if bar_low < lowest_since_entry:
                lowest_since_entry = bar_low
            
            # Update trailing stop
            new_stop = update_trailing_stop(
                current_stop, current_price, entry_price,
                highest_since_entry, lowest_since_entry,
                bias, TRAIL_DISTANCE_PCT, BREAKEVEN_AT_R, TRAIL_ACTIVATION_R
            )
            
            current_trade['current_stop'] = new_stop
            current_trade['highest_since_entry'] = highest_since_entry
            current_trade['lowest_since_entry'] = lowest_since_entry
            
            # Check for exits
            exit_triggered = False
            exit_type = None
            close_price = None
            
            # Check TP
            tp_hit = (bar_high >= tp_price if bias == "bull" else bar_low <= tp_price)
            
            # Check trailing stop
            stop_hit = check_trailing_stop_hit(bar_high, bar_low, new_stop, bias)
            
            if tp_hit and stop_hit:
                # Both hit - conservative: assume stop first
                exit_type = "trail_stop"
                close_price = new_stop * ((1 - SLIPPAGE_PCT) if bias=="bull" else (1 + SLIPPAGE_PCT))
                exit_triggered = True
            elif tp_hit:
                exit_type = "tp"
                close_price = tp_price * ((1 + SLIPPAGE_PCT) if bias=="bull" else (1 - SLIPPAGE_PCT))
                exit_triggered = True
            elif stop_hit:
                exit_type = "trail_stop"
                close_price = new_stop * ((1 - SLIPPAGE_PCT) if bias=="bull" else (1 + SLIPPAGE_PCT))
                exit_triggered = True
            
            # Execute exit
            if exit_triggered:
                pl = (close_price - entry_price) * size if bias=="bull" else (entry_price - close_price) * size
                exit_fee = abs(close_price * size) * TAKER_FEE_RATE
                
                # Update capital
                capital += pl
                capital -= fee_entry
                capital -= exit_fee
                
                # Stats
                if pl > 0:
                    legs_stats["win"] += 1
                else:
                    legs_stats["loss"] += 1
                
                trade_num += 1
                
                # Calculate R-multiple achieved
                initial_risk = abs(entry_price - initial_stop)
                r_achieved = pl / (size * initial_risk) if initial_risk > 0 else 0
                
                results.append({
                    "trade": trade_num,
                    "entry_index": entry_idx,
                    "entry_time": df["time"].iloc[entry_idx],
                    "exit_index": i,
                    "exit_time": df["time"].iloc[i],
                    "duration_bars": i - entry_idx,
                    "bias": bias,
                    "zone": zone,
                    "initial_rr": rr_ratio,
                    "r_achieved": r_achieved,
                    "size": size,
                    "entry": entry_price,
                    "initial_stop": initial_stop,
                    "final_stop": new_stop,
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
                
                # Clear trade
                current_trade = None
            
            # Skip to next bar
            continue
        
        # ===============================================================
        # PART 2: LOOK FOR NEW ENTRY
        # ===============================================================
        
        if i not in entry_signals or i >= len(df) - 1:
            continue
        
        bias = df['bias'].iloc[i]
        
        # Calculate S/R zones
        sr_zones = calculate_sr_for_bar(df, i, SR_LOOKBACK, SR_MIN_TOUCHES)
        
        if sr_zones is None:
            continue
        
        # Check for pullback entry
        if bias == "bull":
            zone_low = sr_zones.get('support_low')
            zone_high = sr_zones.get('support_high')
            if zone_low is None or zone_high is None:
                continue
            zone_type = "support"
        else:  # bear
            zone_low = sr_zones.get('resistance_low')
            zone_high = sr_zones.get('resistance_high')
            if zone_low is None or zone_high is None:
                continue
            zone_type = "resistance"
        
        # Wait for pullback into zone
        if not check_pullback_entry(df, i, zone_low, zone_high, bias, PULLBACK_TO_ZONE_PCT):
            continue
        
        # Entry at next bar open
        entry_price = float(df["open"].iloc[i+1])
        
        # Find swing for stop loss
        if bias == "bull":
            swing_level = find_recent_swing_low(df, i, SWING_LOOKBACK)
        else:
            swing_level = find_recent_swing_high(df, i, SWING_LOOKBACK)
        
        if swing_level is None:
            continue
        
        # Calculate stop with tolerance zone
        stop_price = calculate_swing_stop(entry_price, swing_level, bias, SWING_TOLERANCE_PCT)
        
        # Validate minimum risk
        risk = abs(entry_price - stop_price)
        risk_pct = (risk / entry_price) * 100
        
        if risk_pct < MIN_RISK_PCT:
            continue  # Risk too small
        
        # Calculate take profit
        reward = risk * RR_FOLLOW
        if bias == "bull":
            tp_price = entry_price + reward
        else:
            tp_price = entry_price - reward
        
        # Position sizing
        notional = capital if capital < MAX_MARGIN else MAX_MARGIN
        size = (notional * LEVERAGE) / entry_price
        
        if size <= 0:
            continue
        
        # Entry fee
        entry_notional = entry_price * size
        fee_entry = entry_notional * TAKER_FEE_RATE
        
        # Open trade
        current_trade = {
            'entry_idx': i + 1,
            'entry_price': entry_price,
            'size': size,
            'initial_stop': stop_price,
            'current_stop': stop_price,
            'tp_price': tp_price,
            'bias': bias,
            'fee_entry': fee_entry,
            'zone': zone_type,
            'rr_ratio': RR_FOLLOW,
            'highest_since_entry': entry_price,
            'lowest_since_entry': entry_price
        }
    
    pbar.close()
    
    # Create symbol-specific results directory
    symbol_dir = os.path.join(RESULTS_DIR, symbol)
    os.makedirs(symbol_dir, exist_ok=True)
    
    # Save results
    res_df = pd.DataFrame(results)
    csv_path = os.path.join(symbol_dir, f"{interval}_{symbol}_backtest.csv")
    res_df.to_csv(csv_path, index=False)
    print(f"\nResults for {symbol} saved to: {csv_path}")
    
    # Generate visualizations
    print(f"Generating charts for {symbol}...")
    plot_equity_curve(equity_curve, symbol, INITIAL_CAPITAL, symbol_dir)
    plot_trade_distribution(res_df, symbol, symbol_dir)
    print(f"✅ Charts saved for {symbol}")
    
    # Calculate metrics
    net_pls = res_df["net"].to_numpy(dtype=float) if not res_df.empty else np.array([], dtype=float)
    total_net = capital - INITIAL_CAPITAL
    avg_pl = np.mean(net_pls) if len(net_pls) > 0 else 0.0
    win_rate = np.mean([1 if x > 0 else 0 for x in net_pls]) * 100 if len(net_pls) > 0 else 0.0
    
    equity = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity)
    drawdowns = (running_max - equity) / running_max
    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
    
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
        "Max DD %": f"{max_dd * 100:.2f}"
    }



def backtest_single_symbol(args):
    """Wrapper for parallel processing."""
    symbol, interval, limit, period = args
    try:
        return run_backtest(symbol, interval, limit, period)
    except Exception as e:
        print(f"Error backtesting {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None
