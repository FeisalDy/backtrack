"""
Swing detection module for V2.
Identifies swing highs/lows for stop loss placement.
"""
import numpy as np


def find_recent_swing_low(df, current_idx, lookback=10):
    """
    Find the most recent swing low (area, not a line).
    
    A swing low is a bar where:
    - Low is lower than N bars before and after it
    
    Args:
        df: DataFrame with OHLC data
        current_idx: Current bar index
        lookback: How many bars to look back
        
    Returns:
        float: Swing low price, or None if not found
    """
    if current_idx < lookback + 2:
        return None
    
    start_idx = max(0, current_idx - lookback)
    lows = df['low'].iloc[start_idx:current_idx].values
    
    if len(lows) < 5:
        return None
    
    # Find swing lows (lower than 2 bars on each side)
    swing_lows = []
    for i in range(2, len(lows) - 2):
        if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
            lows[i] < lows[i+1] and lows[i] < lows[i+2]):
            swing_lows.append(lows[i])
    
    if not swing_lows:
        # No clear swing, use recent low
        return float(np.min(lows))
    
    # Return the most recent (last) swing low
    return float(swing_lows[-1])


def find_recent_swing_high(df, current_idx, lookback=10):
    """
    Find the most recent swing high (area, not a line).
    
    A swing high is a bar where:
    - High is higher than N bars before and after it
    
    Args:
        df: DataFrame with OHLC data
        current_idx: Current bar index
        lookback: How many bars to look back
        
    Returns:
        float: Swing high price, or None if not found
    """
    if current_idx < lookback + 2:
        return None
    
    start_idx = max(0, current_idx - lookback)
    highs = df['high'].iloc[start_idx:current_idx].values
    
    if len(highs) < 5:
        return None
    
    # Find swing highs (higher than 2 bars on each side)
    swing_highs = []
    for i in range(2, len(highs) - 2):
        if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
            highs[i] > highs[i+1] and highs[i] > highs[i+2]):
            swing_highs.append(highs[i])
    
    if not swing_highs:
        # No clear swing, use recent high
        return float(np.max(highs))
    
    # Return the most recent (last) swing high
    return float(swing_highs[-1])


def calculate_swing_stop(entry_price, swing_level, bias, tolerance_pct=0.15):
    """
    Calculate stop loss based on swing level with tolerance zone.
    
    The swing is not a line, but an AREA to avoid liquidity hunts.
    
    Args:
        entry_price: Entry price
        swing_level: Swing high/low price
        bias: 'bull' or 'bear'
        tolerance_pct: Tolerance as % (0.15 = 0.15% zone)
        
    Returns:
        float: Stop loss price with tolerance applied
    """
    if bias == "bull":
        # Stop below swing low, with extra tolerance to avoid liquidity hunts
        stop = swing_level * (1 - tolerance_pct / 100)
    else:  # bear
        # Stop above swing high, with extra tolerance
        stop = swing_level * (1 + tolerance_pct / 100)
    
    return stop


def check_pullback_entry(df, current_idx, zone_low, zone_high, bias, pullback_pct=0.5):
    """
    Check if price is near or in the zone for entry.
    
    Simplified: Just check if current price is within reasonable distance of zone.
    
    Args:
        df: DataFrame with OHLC data
        current_idx: Current bar index
        zone_low: Zone bottom
        zone_high: Zone top
        bias: 'bull' or 'bear'
        pullback_pct: Not used anymore - kept for compatibility
        
    Returns:
        bool: True if price is valid for entry
    """
    current_low = float(df['low'].iloc[current_idx])
    current_high = float(df['high'].iloc[current_idx])
    
    if bias == "bull":
        # For bull, accept if price has touched or is near support zone
        # Allow entries even if price is slightly above zone
        zone_top_buffer = zone_high * 1.01  # 1% above zone is OK
        return current_low <= zone_top_buffer
    else:  # bear
        # For bear, accept if price has touched or is near resistance zone
        # Allow entries even if price is slightly below zone
        zone_bottom_buffer = zone_low * 0.99  # 1% below zone is OK
        return current_high >= zone_bottom_buffer
