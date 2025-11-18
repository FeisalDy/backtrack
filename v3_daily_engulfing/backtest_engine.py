"""V3 Backtest Engine"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from config import *
from data_loader import load_from_yfinance
from indicators import compute_indicators
from daily_levels import mark_daily_levels, is_trading_window, is_day_end
from trend_detection import add_trend_to_15m
from pattern_detection import (is_bullish_engulfing, is_bearish_engulfing,
                               check_sweep, detect_bullish_fvg, detect_bearish_fvg)
from visualization import plot_equity_curve, plot_trade_distribution

def run_backtest(symbol, interval, limit, period):
    df_15m = load_from_yfinance(symbol, "15m", limit, period)
    df_15m = compute_indicators(df_15m)
    df_4h = load_from_yfinance(symbol, "4h", limit, period)
    df_4h = compute_indicators(df_4h)
    
    print(f"Marking daily levels for {symbol}...")
    df_15m = mark_daily_levels(df_15m)
    print(f"Detecting 4h trend for {symbol}...")
    df_15m = add_trend_to_15m(df_15m, df_4h, TREND_EMA_FAST, TREND_EMA_SLOW)
    
    capital = INITIAL_CAPITAL
    equity_curve = [capital]
    results = []
    legs_stats = {"win": 0, "loss": 0}
    trade_num = 0
    current_trade = None
    
    pbar = tqdm(total=len(df_15m), desc=f"Backtesting {symbol}", unit="bar", leave=False)
    
    for i in range(len(df_15m)):
        pbar.update(1)
        current_time = df_15m['time'].iloc[i]
        
        if current_trade is not None:
            entry_idx = current_trade['entry_idx']
            entry_price = current_trade['entry_price']
            size = current_trade['size']
            stop_price = current_trade['stop_price']
            tp_price = current_trade['tp_price']
            direction = current_trade['direction']
            fee_entry = current_trade['fee_entry']
            
            bar_high = df_15m['high'].iloc[i]
            bar_low = df_15m['low'].iloc[i]
            
            exit_triggered = False
            exit_type = None
            close_price = None
            
            tp_hit = (bar_high >= tp_price if direction == "long" else bar_low <= tp_price)
            sl_hit = (bar_low <= stop_price if direction == "long" else bar_high >= stop_price)
            day_end = is_day_end(current_time, DAY_END_HOUR, DAY_END_MINUTE)
            
            if tp_hit and sl_hit:
                exit_type = "sl"
                close_price = stop_price * ((1 - SLIPPAGE_PCT) if direction=="long" else (1 + SLIPPAGE_PCT))
                exit_triggered = True
            elif tp_hit:
                exit_type = "tp"
                close_price = tp_price * ((1 + SLIPPAGE_PCT) if direction=="long" else (1 - SLIPPAGE_PCT))
                exit_triggered = True
            elif sl_hit:
                exit_type = "sl"
                close_price = stop_price * ((1 - SLIPPAGE_PCT) if direction=="long" else (1 + SLIPPAGE_PCT))
                exit_triggered = True
            elif day_end:
                exit_type = "day_end"
                close_price = df_15m['close'].iloc[i]
                exit_triggered = True
            
            if exit_triggered:
                pl = (close_price - entry_price) * size if direction=="long" else (entry_price - close_price) * size
                exit_fee = abs(close_price * size) * TAKER_FEE_RATE
                capital += pl - fee_entry - exit_fee
                
                if pl > 0:
                    legs_stats["win"] += 1
                else:
                    legs_stats["loss"] += 1
                
                trade_num += 1
                results.append({
                    "trade": trade_num, "entry_index": entry_idx,
                    "entry_time": df_15m["time"].iloc[entry_idx],
                    "exit_index": i, "exit_time": current_time,
                    "duration_bars": i - entry_idx, "direction": direction,
                    "size": size, "entry": entry_price, "stop_loss": stop_price,
                    "take_profit": tp_price, "close_price": close_price,
                    "exit_type": exit_type, "pl": pl, "fee_entry": fee_entry,
                    "fee_exit": exit_fee, "net": pl - fee_entry - exit_fee,
                    "balance": capital
                })
                equity_curve.append(capital)
                current_trade = None
            continue
        
        if not is_trading_window(current_time, TRADING_START_HOUR, DAY_END_HOUR, DAY_END_MINUTE):
            continue
        
        prev_day_high = df_15m['prev_day_high'].iloc[i]
        prev_day_low = df_15m['prev_day_low'].iloc[i]
        if pd.isna(prev_day_high) or pd.isna(prev_day_low):
            continue
        
        trend_4h = df_15m['trend_4h'].iloc[i]
        if trend_4h == "neutral":
            continue
        
        if trend_4h == "uptrend":
            if not is_bullish_engulfing(df_15m, i, MIN_ENGULFING_SIZE_PCT, MIN_BODY_TO_WICK_RATIO):
                continue
            if not check_sweep(df_15m, i, LOOKBACK_FOR_SWEEP, SWEEP_TOLERANCE_PCT, "low"):
                continue
            fvg = detect_bullish_fvg(df_15m, i, MIN_FVG_SIZE_PCT)
            if fvg is None:
                continue
            direction = "long"
            entry_price = fvg['fvg_mid']
            stop_price = df_15m['low'].iloc[i] * (1 - STOP_BELOW_ENGULFING_PCT / 100)
            tp_price = prev_day_high
        else:
            if not is_bearish_engulfing(df_15m, i, MIN_ENGULFING_SIZE_PCT, MIN_BODY_TO_WICK_RATIO):
                continue
            if not check_sweep(df_15m, i, LOOKBACK_FOR_SWEEP, SWEEP_TOLERANCE_PCT, "high"):
                continue
            fvg = detect_bearish_fvg(df_15m, i, MIN_FVG_SIZE_PCT)
            if fvg is None:
                continue
            direction = "short"
            entry_price = fvg['fvg_mid']
            stop_price = df_15m['high'].iloc[i] * (1 + STOP_BELOW_ENGULFING_PCT / 100)
            tp_price = prev_day_low
        
        risk = abs(entry_price - stop_price)
        risk_pct = (risk / entry_price) * 100
        if risk_pct < MIN_RISK_PCT or risk_pct > MAX_RISK_PCT:
            continue
        
        notional = capital if capital < MAX_MARGIN else MAX_MARGIN
        size = (notional * LEVERAGE) / entry_price
        if size <= 0:
            continue
        
        fee_entry = entry_price * size * TAKER_FEE_RATE
        current_trade = {'entry_idx': i, 'entry_price': entry_price, 'size': size,
                        'stop_price': stop_price, 'tp_price': tp_price,
                        'direction': direction, 'fee_entry': fee_entry}
    
    pbar.close()
    
    symbol_dir = os.path.join(RESULTS_DIR, symbol)
    os.makedirs(symbol_dir, exist_ok=True)
    res_df = pd.DataFrame(results)
    csv_path = os.path.join(symbol_dir, f"{interval}_{symbol}_backtest.csv")
    res_df.to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}")
    
    print(f"Generating charts for {symbol}...")
    plot_equity_curve(equity_curve, symbol, INITIAL_CAPITAL, symbol_dir)
    plot_trade_distribution(res_df, symbol, symbol_dir)
    print(f"✅ Charts saved for {symbol}")
    
    net_pls = res_df["net"].to_numpy(dtype=float) if not res_df.empty else np.array([])
    total_net = capital - INITIAL_CAPITAL
    avg_pl = np.mean(net_pls) if len(net_pls) > 0 else 0.0
    win_rate = np.mean([1 if x > 0 else 0 for x in net_pls]) * 100 if len(net_pls) > 0 else 0.0
    
    equity = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity)
    drawdowns = (running_max - equity) / running_max
    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
    
    return {"Symbol": symbol, "Timeframe": interval,
            "Start Date": df_15m['time'].iloc[0].strftime('%Y-%m-%d'),
            "End Date": df_15m['time'].iloc[-1].strftime('%Y-%m-%d'),
            "Bars": len(df_15m), "Trades": len(res_df),
            "Start Cap": f"${INITIAL_CAPITAL:.2f}", "End Cap": f"${capital:.2f}",
            "Net P/L": f"${total_net:.2f}", "Avg P/L": f"${avg_pl:.2f}",
            "Win Rate %": f"{win_rate:.2f}", "Wins": legs_stats['win'],
            "Losses": legs_stats['loss'], "Max DD %": f"{max_dd * 100:.2f}"}

def backtest_single_symbol(args):
    symbol, interval, limit, period = args
    try:
        return run_backtest(symbol, interval, limit, period)
    except Exception as e:
        print(f"Error backtesting {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None
