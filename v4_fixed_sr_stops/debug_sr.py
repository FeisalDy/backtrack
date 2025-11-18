"""
Debug S/R zone detection.
"""
import pandas as pd
import numpy as np
from data_loader import load_from_yfinance
from indicators import compute_indicators
from support_resistance import calculate_sr_for_bar
from config import SR_LOOKBACK, SR_MIN_TOUCHES, SR_ZONE_TOLERANCE

# Test with BTC-USD
symbol = "BTC-USD"
df = load_from_yfinance(symbol, "5m", 100000, "60d")
df = compute_indicators(df)

# Test zone detection at a specific bar
test_idx = 100

print(f"Testing S/R detection at bar {test_idx}")
print(f"SR_LOOKBACK: {SR_LOOKBACK}")
print(f"SR_MIN_TOUCHES: {SR_MIN_TOUCHES}")
print(f"SR_ZONE_TOLERANCE: {SR_ZONE_TOLERANCE}")
print()

sr_zones = calculate_sr_for_bar(df, test_idx, SR_LOOKBACK, SR_MIN_TOUCHES, SR_ZONE_TOLERANCE)

if sr_zones:
    print("✅ Zones found:")
    print(f"  Support: {sr_zones['support_low']} - {sr_zones['support_high']}")
    print(f"  Resistance: {sr_zones['resistance_low']} - {sr_zones['resistance_high']}")
else:
    print("❌ No zones returned")

# Let's manually check what the algorithm should find
print("\n" + "="*70)
print("MANUAL ZONE DETECTION:")
print("="*70)

recent_high = df['high'].iloc[test_idx-SR_LOOKBACK:test_idx].values
recent_low = df['low'].iloc[test_idx-SR_LOOKBACK:test_idx].values

print(f"\nLooking at {len(recent_low)} bars from {test_idx-SR_LOOKBACK} to {test_idx-1}")
print(f"Low range: ${recent_low.min():.2f} - ${recent_low.max():.2f}")
print(f"High range: ${recent_high.min():.2f} - ${recent_high.max():.2f}")

# Find local lows
local_lows = []
for j in range(1, len(recent_low) - 1):
    if (recent_low[j] <= recent_low[j-1] and 
        recent_low[j] <= recent_low[j+1]):
        local_lows.append(recent_low[j])

# Find local highs
local_highs = []
for j in range(1, len(recent_high) - 1):
    if (recent_high[j] >= recent_high[j-1] and 
        recent_high[j] >= recent_high[j+1]):
        local_highs.append(recent_high[j])

print(f"\nLocal lows found: {len(local_lows)}")
if len(local_lows) > 0:
    print(f"  Min: ${min(local_lows):.2f}, Max: ${max(local_lows):.2f}, Median: ${np.median(local_lows):.2f}")

print(f"\nLocal highs found: {len(local_highs)}")
if len(local_highs) > 0:
    print(f"  Min: ${min(local_highs):.2f}, Max: ${max(local_highs):.2f}, Median: ${np.median(local_highs):.2f}")

print(f"\nSR_MIN_TOUCHES = {SR_MIN_TOUCHES}")
print(f"Need at least {SR_MIN_TOUCHES} touches to create a zone")

if len(local_lows) >= SR_MIN_TOUCHES:
    support_level = np.median(local_lows)
    tolerance = support_level * SR_ZONE_TOLERANCE
    print(f"\n✅ Support zone would be created:")
    print(f"   Level: ${support_level:.2f}")
    print(f"   Zone: ${support_level - tolerance:.2f} - ${support_level + tolerance:.2f}")
else:
    print(f"\n❌ Not enough local lows ({len(local_lows)}) for support zone")

if len(local_highs) >= SR_MIN_TOUCHES:
    resistance_level = np.median(local_highs)
    tolerance = resistance_level * SR_ZONE_TOLERANCE
    print(f"\n✅ Resistance zone would be created:")
    print(f"   Level: ${resistance_level:.2f}")
    print(f"   Zone: ${resistance_level - tolerance:.2f} - ${resistance_level + tolerance:.2f}")
else:
    print(f"\n❌ Not enough local highs ({len(local_highs)}) for resistance zone")
