#!/usr/bin/env python3
"""
Cryptocurrency Historical Data Fetcher using CryptoCompare
Gets multi-year historical data (up to 7 years for hourly data)
Free API - no key required for basic usage
Downloads and saves as Parquet files
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import requests
from datetime import datetime, timedelta
from tqdm import tqdm
import time
import os

# -------------------------
# CONFIGURATION
# -------------------------
SYMBOLS = {
    "BTC": "Bitcoin",
    "ETH": "Ethereum", 
    "DOGE": "Dogecoin",
    "SOL": "Solana",
    "ADA": "Cardano"
}

# Available intervals with max data available:
# '5min': ~7 days (CryptoCompare free tier limit for minute data)
# '15min': ~7 days  
# '30min': ~7 days
# 'hour': ~7 years
# 'day': all history

# Configure which intervals to download
INTERVALS_CONFIG = {
    '5min': 7,       # 7 days of 5-minute data (CryptoCompare limit)
    '15min': 7,      # 7 days of 15-minute data
    '30min': 60,     # 60 days of 30-minute data (we'll try to get more)
    'hour': 1095,    # 3 years of hourly data
}

OUTPUT_DIR = "historical_data"

BASE_URL = "https://min-api.cryptocompare.com/data/v2"

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def ensure_output_dir():
    """Create output directory if it doesn't exist"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

def fetch_historical_data(symbol, interval='hour', limit=2000, to_timestamp=None):
    """
    Fetch historical OHLCV data from CryptoCompare
    
    Args:
        symbol: Crypto symbol (BTC, ETH, etc.)
        interval: '5min', '15min', '30min', 'hour', or 'day'
        limit: Number of data points (max 2000 per request)
        to_timestamp: End timestamp (None = now)
    """
    # Map our interval names to CryptoCompare API endpoints
    # For sub-hourly, we use histominute endpoint with aggregate parameter
    endpoint_map = {
        '5min': f"{BASE_URL}/histominute",
        '15min': f"{BASE_URL}/histominute",
        '30min': f"{BASE_URL}/histominute",
        'hour': f"{BASE_URL}/histohour",
        'day': f"{BASE_URL}/histoday"
    }
    
    # Aggregation values for minute-based intervals
    aggregate_map = {
        '5min': 5,
        '15min': 15,
        '30min': 30,
        'hour': 1,
        'day': 1
    }
    
    endpoint = endpoint_map.get(interval, f"{BASE_URL}/histohour")
    aggregate = aggregate_map.get(interval, 1)
    
    params = {
        'fsym': symbol,
        'tsym': 'USD',
        'limit': min(limit, 2000),  # API max is 2000
    }
    
    # Add aggregate parameter for minute-based intervals
    if interval in ['5min', '15min', '30min']:
        params['aggregate'] = aggregate
    
    if to_timestamp:
        params['toTs'] = int(to_timestamp)
    
    try:
        response = requests.get(endpoint, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data['Response'] == 'Success':
            return data['Data']['Data']
        else:
            print(f"API Error: {data.get('Message', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"Request error: {e}")
        return None

def fetch_all_historical_data(symbol, interval='hour', days=365):
    """
    Fetch all historical data by making multiple API calls
    """
    all_data = []
    
    # Calculate how many API calls we need
    intervals_per_day = {
        '5min': 288,      # 24*60/5 = 288 candles per day
        '15min': 96,      # 24*60/15 = 96 candles per day
        '30min': 48,      # 24*60/30 = 48 candles per day
        'hour': 24,
        'day': 1
    }
    intervals_needed = days * intervals_per_day[interval]
    batch_size = 2000  # Max per API call
    num_batches = (intervals_needed // batch_size) + 1
    
    print(f"Fetching ~{intervals_needed:,} {interval}ly candles in {num_batches} batches...")
    
    # Start from now and work backwards
    to_timestamp = int(datetime.now().timestamp())
    
    pbar = tqdm(total=num_batches, desc=f"Downloading {symbol}", unit="batch")
    
    for batch in range(num_batches):
        data = fetch_historical_data(symbol, interval, batch_size, to_timestamp)
        
        if data is None or len(data) == 0:
            break
        
        all_data.extend(data)
        
        # Update progress
        pbar.update(1)
        pbar.set_postfix({
            'candles': len(all_data),
            'date': datetime.fromtimestamp(data[0]['time']).strftime('%Y-%m-%d')
        })
        
        # Move timestamp to before this batch
        oldest_timestamp = data[0]['time']
        to_timestamp = oldest_timestamp - 1
        
        # Rate limiting - be respectful
        time.sleep(0.2)
        
        # Stop if we've got enough data
        if len(all_data) >= intervals_needed:
            break
    
    pbar.close()
    
    return all_data

def data_to_dataframe(data):
    """
    Convert API data to pandas DataFrame
    """
    df = pd.DataFrame(data)
    
    # Rename columns to match our format
    df = df.rename(columns={
        'time': 'timestamp',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volumefrom': 'volume'
    })
    
    # Create datetime column
    df['time'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Select and reorder columns
    df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
    
    # Remove duplicates and sort
    df = df.drop_duplicates(subset=['time']).sort_values('time').reset_index(drop=True)
    
    return df

def save_to_parquet(df, symbol, interval):
    """
    Save DataFrame to Parquet format
    """
    filename = f"{symbol}_USD_{interval}.parquet"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    df.to_parquet(
        filepath,
        engine='pyarrow',
        compression='snappy',
        index=False
    )
    
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    
    print(f"Saved to: {filepath}")
    print(f"File size: {file_size_mb:.2f} MB")
    
    return filepath

def fetch_symbol_data(symbol, name, interval, days):
    """
    Main function to fetch and save data for one symbol
    """
    print(f"\n{'='*70}")
    print(f"Fetching {name} ({symbol}) - {interval}ly data - {days} days")
    print(f"{'='*70}")
    
    # Fetch data
    data = fetch_all_historical_data(symbol, interval, days)
    
    if not data:
        print(f"No data fetched for {symbol}")
        return None
    
    # Convert to DataFrame
    df = data_to_dataframe(data)
    
    print(f"\nTotal candles: {len(df):,}")
    print(f"Date range: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
    print(f"Actual days: {(df['time'].iloc[-1] - df['time'].iloc[0]).days}")
    
    return df

# -------------------------
# MAIN EXECUTION
# -------------------------
def main():
    ensure_output_dir()
    
    print("="*70)
    print("CRYPTOCURRENCY HISTORICAL DATA FETCHER")
    print("Using CryptoCompare API (Free)")
    print("="*70)
    print(f"Symbols: {', '.join([f'{s} ({n})' for s, n in SYMBOLS.items()])}")
    print(f"Intervals to download:")
    for interval, days in INTERVALS_CONFIG.items():
        print(f"  - {interval}: {days} days")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*70)
    
    results = {}
    
    # Download for each symbol and interval combination
    for symbol, name in SYMBOLS.items():
        results[symbol] = {}
        
        for interval, days in INTERVALS_CONFIG.items():
            try:
                # Fetch data
                df = fetch_symbol_data(symbol, name, interval, days)
                
                if df is not None and len(df) > 0:
                    # Save to parquet
                    filepath = save_to_parquet(df, symbol, interval)
                    
                    results[symbol][interval] = {
                        'success': True,
                        'candles': len(df),
                        'filepath': filepath,
                        'date_range': (df['time'].iloc[0], df['time'].iloc[-1]),
                        'days': (df['time'].iloc[-1] - df['time'].iloc[0]).days
                    }
                else:
                    results[symbol][interval] = {'success': False, 'error': 'No data fetched'}
                    
            except Exception as e:
                print(f"\n✗ Error processing {symbol} {interval}: {e}")
                results[symbol][interval] = {'success': False, 'error': str(e)}
            
            # Delay between requests
            time.sleep(0.5)
        
        # Longer delay between symbols
        time.sleep(1)
    
    # Print summary
    print("\n" + "="*70)
    print("DOWNLOAD SUMMARY")
    print("="*70)
    
    total_success = 0
    total_candles = 0
    total_files = len(SYMBOLS) * len(INTERVALS_CONFIG)
    
    for symbol, intervals_results in results.items():
        print(f"\n{symbol}:")
        for interval, result in intervals_results.items():
            if result['success']:
                total_success += 1
                total_candles += result['candles']
                print(f"  {interval:6s}: ✓ {result['candles']:,} candles | {result['days']} days | {result['filepath']}")
            else:
                print(f"  {interval:6s}: ✗ FAILED - {result.get('error', 'Unknown error')}")
    
    print("\n" + "="*70)
    print(f"DONE! Successfully downloaded {total_success}/{total_files} files")
    print(f"Total candles: {total_candles:,}")
    print("="*70)
    
    if total_success > 0:
        print("\nTo load the data in your scripts, use:")
        print("  import pandas as pd")
        print(f"  # For 5-minute data:")
        print(f"  df = pd.read_parquet('{OUTPUT_DIR}/BTC_USD_5min.parquet')")
        print(f"  # For 15-minute data:")
        print(f"  df = pd.read_parquet('{OUTPUT_DIR}/BTC_USD_15min.parquet')")
        print(f"  # For 30-minute data:")
        print(f"  df = pd.read_parquet('{OUTPUT_DIR}/BTC_USD_30min.parquet')")
        print(f"  # For hourly data:")
        print(f"  df = pd.read_parquet('{OUTPUT_DIR}/BTC_USD_hour.parquet')")
        print("  df['time'] = pd.to_datetime(df['time'])")

if __name__ == "__main__":
    main()
