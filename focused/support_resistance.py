"""
Support and Resistance detection module.
Single Responsibility: Identify support and resistance zones for better entry points.
"""
import numpy as np
import pandas as pd


def find_support_resistance_zones(df, lookback=20, min_touches=2):
    """
    Find support and resistance zones based on price action.
    
    Args:
        df: DataFrame with OHLCV data
        lookback: Number of bars to look back for zone detection
        min_touches: Minimum number of touches to confirm a zone
        
    Returns:
        DataFrame with added 'support_zone_low', 'support_zone_high', 
        'resistance_zone_low', 'resistance_zone_high' columns
    """
    n = len(df)
    
    # Initialize zone arrays
    df['support_zone_low'] = np.nan
    df['support_zone_high'] = np.nan
    df['resistance_zone_low'] = np.nan
    df['resistance_zone_high'] = np.nan
    
    for i in range(lookback, n):
        # Get recent price action
        recent_high = df['high'].iloc[i-lookback:i].values
        recent_low = df['low'].iloc[i-lookback:i].values
        recent_close = df['close'].iloc[i-lookback:i].values
        
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
        
        # Create support zone if we have enough local lows
        if len(local_lows) >= min_touches:
            support_level = np.median(local_lows)
            tolerance = np.std(local_lows) if len(local_lows) > 1 else support_level * 0.01
            df.loc[df.index[i], 'support_zone_low'] = support_level - tolerance
            df.loc[df.index[i], 'support_zone_high'] = support_level + tolerance
        
        # Create resistance zone if we have enough local highs
        if len(local_highs) >= min_touches:
            resistance_level = np.median(local_highs)
            tolerance = np.std(local_highs) if len(local_highs) > 1 else resistance_level * 0.01
            df.loc[df.index[i], 'resistance_zone_low'] = resistance_level - tolerance
            df.loc[df.index[i], 'resistance_zone_high'] = resistance_level + tolerance
    
    return df


def get_entry_from_zone(df, index, bias, zone_penetration=0.2):
    """
    Calculate optimal entry point within support/resistance zone.
    
    For BULL bias:
    - Entry: 20% above support zone low (conservative entry in support)
    - Stop: 20% below support zone low (below the zone)
    
    For BEAR bias:
    - Entry: 20% below resistance zone high (conservative entry in resistance)
    - Stop: 20% above resistance zone high (above the zone)
    
    Args:
        df: DataFrame with S/R zones
        index: Current bar index
        bias: Trade direction ('bull' or 'bear')
        zone_penetration: How deep into zone to enter (0.2 = 20%)
        
    Returns:
        Dictionary with 'entry', 'stop', or None if no valid zone
    """
    if bias == "bull":
        support_low = df.loc[index, 'support_zone_low']
        support_high = df.loc[index, 'support_zone_high']
        
        if pd.isna(support_low) or pd.isna(support_high):
            return None
        
        zone_size = support_high - support_low
        
        # Entry: 20% into the support zone from bottom
        entry = support_low + (zone_size * zone_penetration)
        
        # Stop: 20% below the support zone
        stop = support_low - (zone_size * zone_penetration)
        
        return {'entry': entry, 'stop': stop, 'zone': 'support'}
        
    else:  # bear
        resistance_low = df.loc[index, 'resistance_zone_low']
        resistance_high = df.loc[index, 'resistance_zone_high']
        
        if pd.isna(resistance_low) or pd.isna(resistance_high):
            return None
        
        zone_size = resistance_high - resistance_low
        
        # Entry: 20% into the resistance zone from top
        entry = resistance_high - (zone_size * zone_penetration)
        
        # Stop: 20% above the resistance zone
        stop = resistance_high + (zone_size * zone_penetration)
        
        return {'entry': entry, 'stop': stop, 'zone': 'resistance'}


def validate_risk_reward(entry, stop, bias, min_rr=2.5):
    """
    Calculate take profit based on risk and validate RR ratio.
    
    Args:
        entry: Entry price
        stop: Stop loss price
        bias: Trade direction
        min_rr: Minimum risk-reward ratio required
        
    Returns:
        Dictionary with 'tp', 'risk', 'reward', 'rr_ratio' or None if RR too low
    """
    if bias == "bull":
        risk = entry - stop
        if risk <= 0:
            return None
        reward = risk * min_rr
        tp = entry + reward
    else:  # bear
        risk = stop - entry
        if risk <= 0:
            return None
        reward = risk * min_rr
        tp = entry - reward
    
    rr_ratio = reward / risk if risk > 0 else 0
    
    if rr_ratio < min_rr:
        return None
    
    return {
        'tp': tp,
        'risk': risk,
        'reward': reward,
        'rr_ratio': rr_ratio
    }
