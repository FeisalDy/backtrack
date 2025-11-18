"""
Support and Resistance detection module.
Single Responsibility: Identify support and resistance zones for better entry points.
"""
import numpy as np
import pandas as pd


def calculate_sr_for_bar(df, bar_idx, lookback=20, min_touches=2):
    """
    Calculate S/R zones for a SINGLE bar using only historical data.
    
    This function ensures NO look-ahead bias by only using data from bar_idx-lookback to bar_idx-1.
    
    Args:
        df: DataFrame with OHLCV data
        bar_idx: Current bar index to calculate zones for
        lookback: Number of bars to look back
        min_touches: Minimum touches to confirm a zone
        
    Returns:
        Dictionary with 'support_low', 'support_high', 'resistance_low', 'resistance_high'
        or None if bar_idx < lookback
    """
    if bar_idx < lookback:
        return None
    
    # Get ONLY historical data (bars before current bar)
    recent_high = df['high'].iloc[bar_idx-lookback:bar_idx].values
    recent_low = df['low'].iloc[bar_idx-lookback:bar_idx].values
    
    # Find local lows (potential support)
    local_lows = []
    for j in range(2, len(recent_low) - 2):
        if (recent_low[j] < recent_low[j-1] and 
            recent_low[j] < recent_low[j-2] and
            recent_low[j] < recent_low[j+1] and 
            recent_low[j] < recent_low[j+2]):
            local_lows.append(recent_low[j])
    
    # Find local highs (potential resistance)
    local_highs = []
    for j in range(2, len(recent_high) - 2):
        if (recent_high[j] > recent_high[j-1] and 
            recent_high[j] > recent_high[j-2] and
            recent_high[j] > recent_high[j+1] and 
            recent_high[j] > recent_high[j+2]):
            local_highs.append(recent_high[j])
    
    result = {}
    
    # Create support zone if we have enough local lows
    if len(local_lows) >= min_touches:
        support_level = np.median(local_lows)
        tolerance = np.std(local_lows) if len(local_lows) > 1 else support_level * 0.01
        result['support_low'] = support_level - tolerance
        result['support_high'] = support_level + tolerance
    else:
        result['support_low'] = None
        result['support_high'] = None
    
    # Create resistance zone if we have enough local highs
    if len(local_highs) >= min_touches:
        resistance_level = np.median(local_highs)
        tolerance = np.std(local_highs) if len(local_highs) > 1 else resistance_level * 0.01
        result['resistance_low'] = resistance_level - tolerance
        result['resistance_high'] = resistance_level + tolerance
    else:
        result['resistance_low'] = None
        result['resistance_high'] = None
    
    # CRITICAL: Validate that support is BELOW resistance
    # If zones overlap, they are invalid (support should never be above resistance)
    if (result['support_high'] is not None and 
        result['resistance_low'] is not None and
        result['support_high'] >= result['resistance_low']):
        # Zones overlap - this is illogical, invalidate both
        result['support_low'] = None
        result['support_high'] = None
        result['resistance_low'] = None
        result['resistance_high'] = None
    
    return result
