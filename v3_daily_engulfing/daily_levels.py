"""Daily level marking and time window checks"""
import pandas as pd

def mark_daily_levels(df):
    """Mark previous day high/low at 7am UTC each day"""
    df = df.copy()
    df['prev_day_high'] = pd.NA
    df['prev_day_low'] = pd.NA
    
    current_day_high = None
    current_day_low = None
    prev_day_high = None
    prev_day_low = None
    last_date = None
    
    for i in range(len(df)):
        current_time = df['time'].iloc[i]
        current_date = current_time.date()
        
        if last_date is None or current_date != last_date:
            if last_date is not None:
                prev_day_high = current_day_high
                prev_day_low = current_day_low
            current_day_high = df['high'].iloc[i]
            current_day_low = df['low'].iloc[i]
            last_date = current_date
        else:
            current_day_high = max(current_day_high, df['high'].iloc[i])
            current_day_low = min(current_day_low, df['low'].iloc[i])
        
        if current_time.hour >= 7 and prev_day_high is not None:
            df.loc[df.index[i], 'prev_day_high'] = prev_day_high
            df.loc[df.index[i], 'prev_day_low'] = prev_day_low
    
    return df

def is_trading_window(timestamp, start_hour, end_hour, end_minute):
    """Check if timestamp is within trading window (7am-11:59pm UTC)"""
    hour = timestamp.hour
    minute = timestamp.minute
    
    if hour < start_hour:
        return False
    if hour > end_hour:
        return False
    if hour == end_hour and minute > end_minute:
        return False
    return True

def is_day_end(timestamp, end_hour, end_minute):
    """Check if it's time to close positions (11:59pm UTC)"""
    return timestamp.hour == end_hour and timestamp.minute == end_minute
