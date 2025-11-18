"""
Trading logic module for V4.
Single Responsibility: Handle stop checking logic.

V4 CHANGES:
- Simplified - trade levels now calculated in support_resistance.py
- Only handles stop checking
"""
import numpy as np


def check_stops_vectorized(df_slice, tp_level, sl_level, bias):
    """
    Check if TP or SL was hit in a slice of bars using vectorized operations.
    
    Performance: ~100x faster than iterative loop checking each bar.
    
    Args:
        df_slice: DataFrame slice to check
        tp_level: Take profit price level
        sl_level: Stop loss price level
        bias: Trade direction ('bull' or 'bear')
        
    Returns:
        Tuple of (index, exit_type) where:
            - index: Relative index where stop was hit (-1 if none)
            - exit_type: 'tp', 'sl', or None
    """
    high = df_slice['high'].values
    low = df_slice['low'].values
    
    if bias == "bull":
        tp_hit = high >= tp_level
        sl_hit = low <= sl_level
    else:  # bear
        tp_hit = low <= tp_level
        sl_hit = high >= sl_level
    
    # Find first occurrence of each
    tp_indices = np.where(tp_hit)[0]
    sl_indices = np.where(sl_hit)[0]
    
    if len(tp_indices) == 0 and len(sl_indices) == 0:
        return -1, None  # No hit
    elif len(tp_indices) == 0:
        return sl_indices[0], "sl"
    elif len(sl_indices) == 0:
        return tp_indices[0], "tp"
    else:
        # Both hit - check which came first
        if tp_indices[0] < sl_indices[0]:
            return tp_indices[0], "tp"
        elif sl_indices[0] < tp_indices[0]:
            return sl_indices[0], "sl"
        else:
            # Same bar - conservative assumption: SL hit first
            return sl_indices[0], "sl"
