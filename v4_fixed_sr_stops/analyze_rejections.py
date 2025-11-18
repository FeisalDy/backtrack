"""
Analyze why trades are being rejected.
"""
import pandas as pd
import numpy as np
from data_loader import load_from_yfinance
from indicators import compute_indicators
from bias_scoring import compute_bias_vectorized
from support_resistance import calculate_sr_for_bar, calculate_trade_levels_v4
from config import SR_LOOKBACK, SR_MIN_TOUCHES, SR_ZONE_TOLERANCE, TARGET_RR_RATIO, ZONE_TARGET_PCT, MIN_RISK_PCT

# Test with BTC-USD
symbol = "BTC-USD"
df = load_from_yfinance(symbol, "5m", 100000, "60d")
df = compute_indicators(df)
bias_series = compute_bias_vectorized(df)
df['bias'] = bias_series

# Track rejection reasons
rejection_stats = {
    'no_bias': 0,
    'no_sr_zones': 0,
    'distance_too_far': 0,
    'already_in_zone': 0,
    'no_trade_levels': 0,
    'stop_conflict': 0,
    'min_risk_too_small': 0,
    'valid_trades': 0
}

# Check a sample of signals
entry_indices = df[df['bias'].notna()].index.tolist()
print(f"Total bars: {len(df)}")
print(f"Bars with bias signal: {len(entry_indices)}")
print(f"\nAnalyzing first 100 bias signals...\n")

for idx, i in enumerate(entry_indices[:100]):
    if i >= len(df) - 1 or i < SR_LOOKBACK:
        continue
    
    bias = df['bias'].iloc[i]
    
    # Calculate S/R zones
    sr_zones = calculate_sr_for_bar(df, i, SR_LOOKBACK, SR_MIN_TOUCHES, SR_ZONE_TOLERANCE)
    
    if sr_zones is None:
        rejection_stats['no_sr_zones'] += 1
        continue
    
    current_price = float(df["close"].iloc[i])
    
    # Check zone requirements
    support_zone = {
        'low': sr_zones['support_low'],
        'high': sr_zones['support_high']
    } if sr_zones['support_low'] is not None else None
    
    resistance_zone = {
        'low': sr_zones['resistance_low'],
        'high': sr_zones['resistance_high']
    } if sr_zones['resistance_low'] is not None else None
    
    # Check what zones we have
    if bias == "bull" and resistance_zone is None:
        rejection_stats['no_sr_zones'] += 1
        if idx < 5:
            print(f"Signal {idx+1}: BULL bias but no resistance zone found")
        continue
    
    if bias == "bear" and support_zone is None:
        rejection_stats['no_sr_zones'] += 1
        if idx < 5:
            print(f"Signal {idx+1}: BEAR bias but no support zone found")
        continue
    
    # Check distance
    if bias == "bull" and resistance_zone:
        distance = resistance_zone['low'] - current_price
        max_distance = current_price * 0.10
        if distance > max_distance:
            rejection_stats['distance_too_far'] += 1
            if idx < 5:
                print(f"Signal {idx+1}: BULL - Distance {distance:.2f} > max {max_distance:.2f}")
            continue
        if current_price >= resistance_zone['low']:
            rejection_stats['already_in_zone'] += 1
            if idx < 5:
                print(f"Signal {idx+1}: BULL - Already in resistance zone")
            continue
    
    if bias == "bear" and support_zone:
        distance = current_price - support_zone['high']
        max_distance = current_price * 0.10
        if distance > max_distance:
            rejection_stats['distance_too_far'] += 1
            if idx < 5:
                print(f"Signal {idx+1}: BEAR - Distance {distance:.2f} > max {max_distance:.2f}")
            continue
        if current_price <= support_zone['high']:
            rejection_stats['already_in_zone'] += 1
            if idx < 5:
                print(f"Signal {idx+1}: BEAR - Already in support zone")
            continue
    
    # Try to calculate trade levels
    trade_levels = calculate_trade_levels_v4(
        current_price=current_price,
        bias=bias,
        support_zone=support_zone,
        resistance_zone=resistance_zone,
        target_rr=TARGET_RR_RATIO,
        zone_target_pct=ZONE_TARGET_PCT,
        min_risk_pct=MIN_RISK_PCT
    )
    
    if trade_levels is None:
        rejection_stats['no_trade_levels'] += 1
        if idx < 5:
            print(f"Signal {idx+1}: Trade levels returned None (likely stop conflict or min risk)")
    else:
        rejection_stats['valid_trades'] += 1
        if idx < 5:
            print(f"Signal {idx+1}: ✅ VALID TRADE - {bias} at ${current_price:.2f}")

print("\n" + "="*70)
print("REJECTION STATISTICS (first 100 signals):")
print("="*70)
for reason, count in rejection_stats.items():
    print(f"{reason:25s}: {count:3d} ({count/100*100:.1f}%)")
print("="*70)
