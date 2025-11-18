"""
Bias scoring module for V5 (bias_v8 logic - ATR filter removed).
Single Responsibility: Calculate bias score based on multiple technical indicators.

Modified from bias_v8_BEST_RESULT.py:
- ATR volatility filter removed to allow trading in all market conditions
- All other scoring components remain the same
"""
import pandas as pd
import numpy as np
from config import BIAS_BULL_THRESHOLD, BIAS_BEAR_THRESHOLD


def compute_bias_for_bar(df, i):
    """
    Compute bias score for a single bar using bias_v8 logic.
    
    Scoring system (max ~20 points in each direction):
    1. Price position in range (±2.5)
    2. Multi-timeframe breakout (±4.0)
    3. Price momentum (±3.0)
    4. Volume confirmation (±2.5)
    5. OBV momentum (±2.0)
    6. Trend strength (±2.0)
    7. MA confirmation (±1.5)
    
    Args:
        df: DataFrame with all indicators
        i: Current bar index
        
    Returns:
        "bull", "bear", or None based on score thresholds
    """
    if i < 100:
        return None

    # Check for NaN values in required columns
    required_cols = [
        'price_in_range_5', 'price_in_range_20', 'recent_high_5', 'recent_low_5',
        'recent_high_20', 'recent_low_20', 'volume_surge',
        'price_momentum_3_zscore', 'price_momentum_10_zscore', 'volume_surge_zscore',
        'obv_momentum_10_zscore', 'obv_momentum_14_zscore', 'trend_strength',
        'fast_ma', 'slow_ma', 'medium_ma', 'close'
    ]
    
    for col in required_cols:
        if pd.isna(df[col].iloc[i]):
            return None

    # ATR volatility filter removed - trade in all market conditions
    
    score = 0.0
    
    # 1. Price position in range
    price_range_5 = df['price_in_range_5'].iloc[i]
    if price_range_5 > 0.75:
        score += 2.5
    elif price_range_5 < 0.25:
        score -= 2.5
    elif price_range_5 > 0.6:
        score += 1.5
    elif price_range_5 < 0.4:
        score -= 1.5

    # 2. Multi-timeframe breakout
    short_breakout_bull = df['close'].iloc[i] > df['recent_high_5'].iloc[i-1]
    short_breakout_bear = df['close'].iloc[i] < df['recent_low_5'].iloc[i-1]
    medium_trend_bull = df['close'].iloc[i] > df['recent_high_20'].iloc[i-1] * 0.995
    medium_trend_bear = df['close'].iloc[i] < df['recent_low_20'].iloc[i-1] * 1.005
    
    if short_breakout_bull and medium_trend_bull:
        score += 4.0
    elif short_breakout_bear and medium_trend_bear:
        score -= 4.0
    elif short_breakout_bull:
        score += 1.5
    elif short_breakout_bear:
        score -= 1.5

    # 3. Price momentum
    momentum_3_z = df['price_momentum_3_zscore'].iloc[i]
    momentum_10_z = df['price_momentum_10_zscore'].iloc[i]
    
    if momentum_3_z > 1.0 and momentum_10_z > 0.5:
        score += 3.0
    elif momentum_3_z < -1.0 and momentum_10_z < -0.5:
        score -= 3.0
    elif momentum_3_z > 0.5:
        score += 1.5
    elif momentum_3_z < -0.5:
        score -= 1.5

    # 4. Volume confirmation
    volume_z = df['volume_surge_zscore'].iloc[i]
    price_up = df['price_momentum_3_zscore'].iloc[i] > 0.5
    price_down = df['price_momentum_3_zscore'].iloc[i] < -0.5
    
    if volume_z > 1.5:
        if price_up:
            score += 2.5
        elif price_down:
            score -= 2.5
    elif volume_z > 1.0:
        if price_up:
            score += 1.5
        elif price_down:
            score -= 1.5

    # 5. OBV momentum
    obv_10_z = df['obv_momentum_10_zscore'].iloc[i]
    obv_14_z = df['obv_momentum_14_zscore'].iloc[i]
    
    if obv_10_z > 0.5 and obv_14_z > 0.5:
        score += 2.0
    elif obv_10_z < -0.5 and obv_14_z < -0.5:
        score -= 2.0
    elif obv_10_z > 1.0 or obv_14_z > 1.0:
        score += 1.0
    elif obv_10_z < -1.0 or obv_14_z < -1.0:
        score -= 1.0

    # 6. Trend strength
    trend_str = df['trend_strength'].iloc[i]
    if abs(trend_str) > 2.0:
        if trend_str > 0:
            score += 2.0
        else:
            score -= 2.0
    elif abs(trend_str) > 1.0:
        if trend_str > 0:
            score += 1.0
        else:
            score -= 1.0

    # 7. MA confirmation
    if df['fast_ma'].iloc[i] > df['slow_ma'].iloc[i] > df['medium_ma'].iloc[i]:
        score += 1.5
    elif df['fast_ma'].iloc[i] < df['slow_ma'].iloc[i] < df['medium_ma'].iloc[i]:
        score -= 1.5

    # Bias determination
    if score >= BIAS_BULL_THRESHOLD:
        return "bull"
    elif score <= BIAS_BEAR_THRESHOLD:
        return "bear"
    else:
        return None


def compute_bias_vectorized(df):
    """
    Compute bias for all bars in the dataframe.
    
    Note: This is not truly vectorized due to complex conditional logic,
    but provides a consistent interface with v1/v4.
    
    Args:
        df: DataFrame with all indicators
        
    Returns:
        Series with bias values ('bull', 'bear', or None)
    """
    bias_list = []
    for i in range(len(df)):
        bias_list.append(compute_bias_for_bar(df, i))
    
    return pd.Series(bias_list, index=df.index)
