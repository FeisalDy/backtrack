"""Pattern detection: engulfing, sweeps, FVG"""
import pandas as pd

def is_bullish_engulfing(df, idx, min_size_pct, min_body_ratio):
    """Check if current candle is bullish engulfing"""
    if idx < 1:
        return False
    
    prev = df.iloc[idx - 1]
    curr = df.iloc[idx]
    
    prev_close = prev['close']
    prev_open = prev['open']
    curr_close = curr['close']
    curr_open = curr['open']
    
    if curr_close <= curr_open:
        return False
    
    if not (curr_open < prev_close and curr_close > prev_open):
        return False
    
    curr_body = curr_close - curr_open
    curr_range = curr['high'] - curr['low']
    size_pct = (curr_range / curr_open) * 100
    
    if size_pct < min_size_pct:
        return False
    
    if curr_range > 0:
        body_ratio = curr_body / curr_range
        if body_ratio < min_body_ratio:
            return False
    
    return True

def is_bearish_engulfing(df, idx, min_size_pct, min_body_ratio):
    """Check if current candle is bearish engulfing"""
    if idx < 1:
        return False
    
    prev = df.iloc[idx - 1]
    curr = df.iloc[idx]
    
    prev_close = prev['close']
    prev_open = prev['open']
    curr_close = curr['close']
    curr_open = curr['open']
    
    if curr_close >= curr_open:
        return False
    
    if not (curr_open > prev_close and curr_close < prev_open):
        return False
    
    curr_body = curr_open - curr_close
    curr_range = curr['high'] - curr['low']
    size_pct = (curr_range / curr_open) * 100
    
    if size_pct < min_size_pct:
        return False
    
    if curr_range > 0:
        body_ratio = curr_body / curr_range
        if body_ratio < min_body_ratio:
            return False
    
    return True

def check_sweep(df, idx, lookback, tolerance_pct, sweep_type):
    """Check if engulfing swept previous low/high"""
    if idx < lookback:
        return False
    
    curr_high = df['high'].iloc[idx]
    curr_low = df['low'].iloc[idx]
    
    window = df.iloc[idx - lookback:idx]
    
    if sweep_type == "low":
        prev_low = window['low'].min()
        tolerance = prev_low * (tolerance_pct / 100)
        if curr_low <= prev_low + tolerance:
            return True
    else:
        prev_high = window['high'].max()
        tolerance = prev_high * (tolerance_pct / 100)
        if curr_high >= prev_high - tolerance:
            return True
    
    return False

def detect_bullish_fvg(df, idx, min_fvg_pct):
    """Detect fair value gap after bullish engulfing"""
    if idx < 2:
        return None
    
    bar_2 = df.iloc[idx - 2]
    bar_0 = df.iloc[idx]
    
    gap_bottom = bar_0['low']
    gap_top = bar_2['high']
    
    if gap_bottom <= gap_top:
        return None
    
    gap_size = gap_bottom - gap_top
    gap_size_pct = (gap_size / bar_0['close']) * 100
    
    if gap_size_pct < min_fvg_pct:
        return None
    
    return {'fvg_top': gap_bottom, 'fvg_bottom': gap_top,
            'fvg_mid': (gap_bottom + gap_top) / 2, 'fvg_size_pct': gap_size_pct}

def detect_bearish_fvg(df, idx, min_fvg_pct):
    """Detect fair value gap after bearish engulfing"""
    if idx < 2:
        return None
    
    bar_2 = df.iloc[idx - 2]
    bar_0 = df.iloc[idx]
    
    gap_top = bar_0['high']
    gap_bottom = bar_2['low']
    
    if gap_top <= gap_bottom:
        return None
    
    gap_size = gap_top - gap_bottom
    gap_size_pct = (gap_size / bar_0['close']) * 100
    
    if gap_size_pct < min_fvg_pct:
        return None
    
    return {'fvg_top': gap_top, 'fvg_bottom': gap_bottom,
            'fvg_mid': (gap_top + gap_bottom) / 2, 'fvg_size_pct': gap_size_pct}
