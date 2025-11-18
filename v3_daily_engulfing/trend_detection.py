"""4H trend detection and mapping to 15m bars"""
import pandas as pd
import numpy as np

def calculate_ema(series, period):
    """Calculate EMA for given period"""
    return series.ewm(span=period, adjust=False).mean()

def add_trend_to_15m(df_15m, df_4h, ema_fast, ema_slow):
    """Map 4h trend to 15m bars"""
    df_15m = df_15m.copy()
    df_4h = df_4h.copy()
    
    df_4h['ema_fast'] = calculate_ema(df_4h['close'], ema_fast)
    df_4h['ema_slow'] = calculate_ema(df_4h['close'], ema_slow)
    
    def get_trend(row):
        close = row['close']
        fast = row['ema_fast']
        slow = row['ema_slow']
        
        if pd.isna(fast) or pd.isna(slow):
            return "neutral"
        
        if close > fast and fast > slow:
            return "uptrend"
        elif close < fast and fast < slow:
            return "downtrend"
        else:
            return "neutral"
    
    df_4h['trend'] = df_4h.apply(get_trend, axis=1)
    
    df_15m['trend_4h'] = "neutral"
    for i in range(len(df_15m)):
        time_15m = df_15m['time'].iloc[i]
        matching_4h = df_4h[df_4h['time'] <= time_15m]
        if not matching_4h.empty:
            latest_trend = matching_4h.iloc[-1]['trend']
            df_15m.loc[df_15m.index[i], 'trend_4h'] = latest_trend
    
    return df_15m
