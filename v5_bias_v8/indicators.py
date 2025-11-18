"""
Technical indicators module for V5 (bias_v8 logic).
Single Responsibility: Calculate all technical indicators needed for bias scoring.
"""
import pandas as pd
import numpy as np


def compute_indicators(df):
    """
    Compute all technical indicators needed for bias_v8 scoring system.
    
    Indicators:
    - Moving averages (fast, slow, medium)
    - OBV (On-Balance Volume)
    - ATR (Average True Range)
    - Price momentum
    - Recent high/low breakouts
    - Volume surge
    - Price position in range
    - OBV momentum
    - Z-score normalization
    - Trend strength
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with all indicators added
    """
    # Moving Averages
    df["fast_ma"] = df["close"].rolling(20).mean()
    df["slow_ma"] = df["close"].rolling(50).mean()
    df["medium_ma"] = df["close"].rolling(100).mean()
    
    # OBV (On-Balance Volume)
    df["delta_close"] = df["close"].diff()
    df["obv"] = 0.0
    obv_values = np.zeros(len(df))
    for i in range(1, len(df)):
        if df["delta_close"].iloc[i] > 0:
            obv_values[i] = obv_values[i-1] + df["volume"].iloc[i]
        elif df["delta_close"].iloc[i] < 0:
            obv_values[i] = obv_values[i-1] - df["volume"].iloc[i]
        else:
            obv_values[i] = obv_values[i-1]
    df["obv"] = obv_values
    
    # ATR (Average True Range)
    df["tr"] = df["high"] - df["low"]
    df["atr"] = df["tr"].rolling(14).mean()
    df["atr_median"] = df["atr"].rolling(50).median()
    
    # Price Momentum
    df['price_momentum_3'] = (df['close'] - df['close'].shift(3)) / df['close'].shift(3) * 100
    df['price_momentum_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100
    
    # Recent High/Low Breakout
    df['recent_high_5'] = df['high'].rolling(5).max()
    df['recent_low_5'] = df['low'].rolling(5).min()
    df['recent_high_20'] = df['high'].rolling(20).max()
    df['recent_low_20'] = df['low'].rolling(20).min()
    
    # Volume Surge
    volume_avg_10 = df['volume'].rolling(10).mean()
    df['volume_surge'] = df['volume'] / (volume_avg_10 + 1e-8)
    
    # Price vs Recent Range
    range_5 = df['recent_high_5'] - df['recent_low_5']
    range_20 = df['recent_high_20'] - df['recent_low_20']
    df['price_in_range_5'] = (df['close'] - df['recent_low_5']) / (range_5 + 1e-8)
    df['price_in_range_20'] = (df['close'] - df['recent_low_20']) / (range_20 + 1e-8)
    
    # OBV Momentum
    df['obv_momentum_10'] = df['obv'].diff(10)
    df['obv_momentum_14'] = df['obv'].diff(14)
    
    # Z-score normalization for key metrics
    for col in ['price_momentum_3', 'price_momentum_10', 'volume_surge', 'obv_momentum_10', 'obv_momentum_14']:
        rolling_mean = df[col].rolling(50).mean()
        rolling_std = df[col].rolling(50).std()
        df[f'{col}_zscore'] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
    
    # Trend strength
    df['trend_strength'] = (df['fast_ma'] - df['slow_ma']) / (df['atr'] + 1e-8)
    
    return df
