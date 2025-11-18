"""
Trailing stop module for V2.
Manages dynamic trailing stops that lock in profits.
"""


def update_trailing_stop(current_stop, current_price, entry_price, highest_price, lowest_price,
                         bias, trail_distance_pct=0.5, breakeven_at_r=0.5, trail_activation_r=1.0):
    """
    Update trailing stop based on current price movement.
    
    Logic:
    1. Move to breakeven after BREAKEVEN_AT_R profit
    2. Start trailing after TRAIL_ACTIVATION_R profit  
    3. Trail at TRAIL_DISTANCE_PCT below high (bull) or above low (bear)
    
    Args:
        current_stop: Current stop loss price
        current_price: Current market price
        entry_price: Entry price
        highest_price: Highest price since entry (for bull trades)
        lowest_price: Lowest price since entry (for bear trades)
        bias: 'bull' or 'bear'
        trail_distance_pct: Trail distance as %
        breakeven_at_r: Move to breakeven after this many R
        trail_activation_r: Start trailing after this many R
        
    Returns:
        float: New stop loss price
    """
    initial_risk = abs(entry_price - current_stop)
    
    if bias == "bull":
        # Bull trade
        profit = current_price - entry_price
        r_multiple = profit / initial_risk if initial_risk > 0 else 0
        
        # Step 1: Move to breakeven after BREAKEVEN_AT_R
        if r_multiple >= breakeven_at_r and current_stop < entry_price:
            current_stop = entry_price
        
        # Step 2: Start trailing after TRAIL_ACTIVATION_R
        if r_multiple >= trail_activation_r:
            # Trail at TRAIL_DISTANCE_PCT below highest price
            trail_stop = highest_price * (1 - trail_distance_pct / 100)
            
            # Only move stop UP, never down
            if trail_stop > current_stop:
                current_stop = trail_stop
    
    else:  # bear
        # Bear trade
        profit = entry_price - current_price
        r_multiple = profit / initial_risk if initial_risk > 0 else 0
        
        # Step 1: Move to breakeven after BREAKEVEN_AT_R
        if r_multiple >= breakeven_at_r and current_stop > entry_price:
            current_stop = entry_price
        
        # Step 2: Start trailing after TRAIL_ACTIVATION_R
        if r_multiple >= trail_activation_r:
            # Trail at TRAIL_DISTANCE_PCT above lowest price
            trail_stop = lowest_price * (1 + trail_distance_pct / 100)
            
            # Only move stop DOWN, never up
            if trail_stop < current_stop:
                current_stop = trail_stop
    
    return current_stop


def check_trailing_stop_hit(bar_high, bar_low, current_stop, bias):
    """
    Check if trailing stop was hit in current bar.
    
    Args:
        bar_high: High of current bar
        bar_low: Low of current bar
        current_stop: Current trailing stop price
        bias: 'bull' or 'bear'
        
    Returns:
        bool: True if stop was hit
    """
    if bias == "bull":
        # Bull: stop hit if low touches stop
        return bar_low <= current_stop
    else:  # bear
        # Bear: stop hit if high touches stop
        return bar_high >= current_stop
