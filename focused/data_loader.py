"""
Data loading module with caching.
Loads OHLCV from Yahoo Finance and caches locally in Parquet.
Cache expires after 1 hour.
"""
import os
import time
import pandas as pd
import yfinance as yf

# Create cache directory relative to this file's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, "data_cache")
CACHE_EXPIRY = 3600  # seconds = 1 hour

def load_from_yfinance(symbol, interval, limit):
    """
    Load historical OHLCV data with caching.
    
    Args:
        symbol: Trading pair symbol (e.g., "BTC-USD")
        interval: Timeframe (e.g., "1h", "5m")
        limit: Maximum number of bars to retrieve
        
    Returns:
        DataFrame with columns: time, open, high, low, close, volume
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_{interval}.parquet")
    fetch_new = True

    # Check if cache exists and is recent
    if os.path.exists(cache_file):
        file_age = time.time() - os.path.getmtime(cache_file)
        if file_age <= CACHE_EXPIRY:
            df = pd.read_parquet(cache_file)
            print(f"Loaded {symbol} {interval} data from cache ({len(df)} bars)")
            fetch_new = False
        else:
            print(f"Cache for {symbol} {interval} expired ({file_age/60:.1f} min old). Fetching new data.")

    if fetch_new:
        print(f"Fetching {symbol} {interval} data from Yahoo Finance...")
        df = yf.download(symbol, period="max", interval=interval, progress=False)

        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Standardize column names
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].rename(
            columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }
        )

        # Add time column
        df = df.reset_index()
        df['time'] = pd.to_datetime(df['Datetime'] if 'Datetime' in df.columns else df['Date'])
        df = df.drop(columns=['Datetime'] if 'Datetime' in df.columns else ['Date'])

        # Optimize memory
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].astype('float32')
        df['volume'] = df['volume'].astype('int64')

        # Save cache
        df.to_parquet(cache_file)
        print(f"Saved {symbol} {interval} data to cache ({len(df)} bars)")

    # Limit the number of bars returned
    if len(df) > limit:
        df = df.tail(limit).reset_index(drop=True)
        print(f"Using last {limit:,} bars for backtest")

    return df
