"""
Support and Resistance detection module for V4.
Single Responsibility: Identify support and resistance zones and calculate trade levels.

V4 CHANGES:
- Target zones (not lines) for take profit
- Fixed stop loss based on risk-reward ratio from target zone
- BULL: Target = nearest resistance zone, Stop = calculated from RR ratio
- BEAR: Target = nearest support zone, Stop = calculated from RR ratio
"""
import numpy as np
import pandas as pd


def calculate_sr_for_bar(df, bar_idx, lookback=30, min_touches=2, zone_tolerance_pct=0.01):
    """
    Calculate S/R zones for a SINGLE bar using only historical data.
    
    This function ensures NO look-ahead bias by only using data from bar_idx-lookback to bar_idx-1.
    
    Args:
        df: DataFrame with OHLCV data
        bar_idx: Current bar index to calculate zones for
        lookback: Number of bars to look back
        min_touches: Minimum touches to confirm a zone
        zone_tolerance_pct: Zone thickness as % of price
        
    Returns:
        Dictionary with 'support_low', 'support_high', 'resistance_low', 'resistance_high'
        or None if bar_idx < lookback
    """
    if bar_idx < lookback:
        return None
    
    # Get ONLY historical data (bars before current bar)
    recent_high = df['high'].iloc[bar_idx-lookback:bar_idx].values
    recent_low = df['low'].iloc[bar_idx-lookback:bar_idx].values
    
    # Current price for reference (use the close before current bar)
    current_price = df['close'].iloc[bar_idx-1]
    
    # Find ALL local lows (swing lows) - don't filter by price yet
    local_lows = []
    for j in range(1, len(recent_low) - 1):  
        if (recent_low[j] <= recent_low[j-1] and 
            recent_low[j] <= recent_low[j+1]):
            local_lows.append(recent_low[j])
    
    # Find ALL local highs (swing highs) - don't filter by price yet
    local_highs = []
    for j in range(1, len(recent_high) - 1):  
        if (recent_high[j] >= recent_high[j-1] and 
            recent_high[j] >= recent_high[j+1]):
            local_highs.append(recent_high[j])
    
    #  Filter to get support BELOW and resistance ABOVE current price
    lows_below = [low for low in local_lows if low < current_price]  
    highs_above = [high for high in local_highs if high > current_price]
    
    result = {}
    
    # Create support zone from lows BELOW current price
    if len(lows_below) >= min_touches:
        support_level = np.median(lows_below)
        # Use fixed % tolerance for zone thickness
        zone_thickness = support_level * zone_tolerance_pct
        result['support_low'] = support_level - zone_thickness
        result['support_high'] = support_level + zone_thickness
    else:
        result['support_low'] = None
        result['support_high'] = None
    
    # Create resistance zone from highs ABOVE current price
    if len(highs_above) >= min_touches:
        resistance_level = np.median(highs_above)
        # Use fixed % tolerance for zone thickness
        zone_thickness = resistance_level * zone_tolerance_pct
        result['resistance_low'] = resistance_level - zone_thickness
        result['resistance_high'] = resistance_level + zone_thickness
    else:
        result['resistance_low'] = None
        result['resistance_high'] = None
    
    # No need to validate overlap - we filtered by price already
    
    return result


def calculate_trade_levels_v4(current_price, bias, support_zone, resistance_zone, 
                               target_rr=2.0, zone_approach_pct=0.3, 
                               zone_target_pct=0.5, min_risk_pct=0.005):
    """
    Calculate entry, stop loss, and take profit for V4 strategy.
    
    V4 LOGIC:
    - BULLISH: 
        * Target = resistance zone (take profit in middle of zone)
        * Entry = current price (approaching resistance)
        * Stop = calculated backward from target using RR ratio
    
    - BEARISH:
        * Target = support zone (take profit in middle of zone)
        * Entry = current price (approaching support)
        * Stop = calculated backward from target using RR ratio
    
    Args:
        current_price: Current market price
        bias: Trade direction ('bull' or 'bear')
        support_zone: Dict with 'low' and 'high' for support zone
        resistance_zone: Dict with 'low' and 'high' for resistance zone
        target_rr: Risk-reward ratio
        zone_approach_pct: How close to zone to consider "approaching"
        zone_target_pct: Where in target zone to place TP (0.5 = middle)
        min_risk_pct: Minimum risk as % of entry price
        
    Returns:
        Dictionary with 'entry', 'stop', 'tp', 'risk', 'reward', 'rr_ratio', 'target_zone'
        or None if no valid setup
    """
    if bias == "bull":
        # Check if we have a valid resistance zone ahead
        if resistance_zone is None or resistance_zone['low'] is None:
            return None
        
        res_low = resistance_zone['low']
        res_high = resistance_zone['high']
        res_mid = (res_low + res_high) / 2
        
        # Check if price is BELOW resistance (approaching from below)
        if current_price >= res_low:
            return None  # Already in or above resistance zone
        
        # Entry at current price (market order)
        entry = current_price
        
        # Take profit in middle of resistance zone (or custom %)
        zone_size = res_high - res_low
        tp = res_low + (zone_size * zone_target_pct)
        
        # Calculate reward
        reward = tp - entry
        
        # Calculate required risk based on RR ratio
        risk = reward / target_rr
        
        # Validate minimum risk
        risk_pct = (risk / entry) * 100
        if risk_pct < min_risk_pct:
            return None
        
        # Stop loss placed backward from entry
        stop = entry - risk
        
        return {
            'entry': entry,
            'stop': stop,
            'tp': tp,
            'risk': risk,
            'reward': reward,
            'rr_ratio': target_rr,
            'target_zone': 'resistance',
            'target_low': res_low,
            'target_high': res_high
        }
        
    else:  # bear
        # Check if we have a valid support zone ahead
        if support_zone is None or support_zone['low'] is None:
            return None
        
        sup_low = support_zone['low']
        sup_high = support_zone['high']
        sup_mid = (sup_low + sup_high) / 2
        
        # Check if price is ABOVE support (approaching from above)
        if current_price <= sup_high:
            return None  # Already in or below support zone
        
        # Entry at current price (market order)
        entry = current_price
        
        # Take profit in middle of support zone (or custom %)
        zone_size = sup_high - sup_low
        tp = sup_high - (zone_size * zone_target_pct)
        
        # Calculate reward
        reward = entry - tp
        
        # Calculate required risk based on RR ratio
        risk = reward / target_rr
        
        # Validate minimum risk
        risk_pct = (risk / entry) * 100
        if risk_pct < min_risk_pct:
            return None
        
        # Stop loss placed backward from entry
        stop = entry + risk
        
        return {
            'entry': entry,
            'stop': stop,
            'tp': tp,
            'risk': risk,
            'reward': reward,
            'rr_ratio': target_rr,
            'target_zone': 'support',
            'target_low': sup_low,
            'target_high': sup_high
        }


def is_approaching_zone(current_price, zone_low, zone_high, bias, approach_pct=0.3):
    """
    Check if price is approaching a target zone.
    
    Args:
        current_price: Current market price
        zone_low: Zone lower boundary
        zone_high: Zone upper boundary
        bias: Trade direction ('bull' or 'bear')
        approach_pct: How close to zone edge to consider "approaching"
        
    Returns:
        Boolean indicating if approaching
    """
    if zone_low is None or zone_high is None:
        return False
    
    zone_size = zone_high - zone_low
    approach_distance = zone_size * approach_pct
    
    if bias == "bull":
        # For bullish, approaching resistance from below
        return zone_low - current_price <= approach_distance and current_price < zone_low
    else:
        # For bearish, approaching support from above
        return current_price - zone_high <= approach_distance and current_price > zone_high
