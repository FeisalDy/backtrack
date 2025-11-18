"""
Technical indicators module.
Single Responsibility: Calculate all technical indicators for the strategy.
"""
import pandas as pd


def compute_indicators(df):
    """
    Compute all technical indicators needed for bias calculation.
    All calculations are fully vectorized for optimal performance.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added indicator columns
    """
    
    # Moving Averages
    df["fast_ma"] = df["close"].rolling(20).mean()
    df["slow_ma"] = df["close"].rolling(50).mean()
    df["medium_ma"] = df["close"].rolling(100).mean()
    
    # On-Balance Volume (OBV) - Vectorized calculation
    delta_close = df["close"].diff()
    volume_signed = df["volume"].copy()
    volume_signed[delta_close < 0] = -volume_signed[delta_close < 0]
    volume_signed[delta_close == 0] = 0
    df["obv"] = volume_signed.cumsum()
    
    # Average True Range (ATR)
    df["tr"] = df["high"] - df["low"]
    df["atr"] = df["tr"].rolling(14).mean()
    df["atr_median"] = df["atr"].rolling(50).median()
    
    # Price Momentum
    df['price_momentum_3'] = (df['close'] - df['close'].shift(3)) / df['close'].shift(3) * 100
    df['price_momentum_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100
    
    # Recent High/Low Breakout levels
    df['recent_high_5'] = df['high'].rolling(5).max()
    df['recent_low_5'] = df['low'].rolling(5).min()
    df['recent_high_20'] = df['high'].rolling(20).max()
    df['recent_low_20'] = df['low'].rolling(20).min()
    
    # Volume Surge
    volume_avg_10 = df['volume'].rolling(10).mean()
    df['volume_surge'] = df['volume'] / volume_avg_10
    
    # Price position in recent range
    df['price_in_range_5'] = (df['close'] - df['recent_low_5']) / (df['recent_high_5'] - df['recent_low_5'])
    df['price_in_range_20'] = (df['close'] - df['recent_low_20']) / (df['recent_high_20'] - df['recent_low_20'])
    
    # OBV Momentum
    df['obv_momentum_10'] = df['obv'].diff(10)
    df['obv_momentum_14'] = df['obv'].diff(14)
    
    # Z-score normalization for key indicators
    for col in ['price_momentum_3', 'price_momentum_10', 'volume_surge', 'obv_momentum_10', 'obv_momentum_14']:
        rolling_mean = df[col].rolling(50).mean()
        rolling_std = df[col].rolling(50).std()
        df[f'{col}_zscore'] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
    
    # Trend strength
    df['trend_strength'] = (df['fast_ma'] - df['slow_ma']) / (df['atr'] + 1e-8)
    
    # Optimize memory: downcast float64 to float32
    for col in df.columns:
        if col not in ['time', 'volume'] and df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
    
    return df
