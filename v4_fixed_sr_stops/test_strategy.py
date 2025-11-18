"""
Test script to validate V4 strategy logic.
"""
import numpy as np
from support_resistance import calculate_trade_levels_v4

print("=" * 70)
print("V4 STRATEGY LOGIC VALIDATION")
print("=" * 70)

# Test Case 1: Bullish Trade
print("\n📈 TEST 1: BULLISH TRADE")
print("-" * 70)
current_price = 103.0  # Close to resistance zone (within 5%)
support_zone = {'low': 90.0, 'high': 92.0}
resistance_zone = {'low': 105.0, 'high': 107.0}

result = calculate_trade_levels_v4(
    current_price=current_price,
    bias='bull',
    support_zone=support_zone,
    resistance_zone=resistance_zone,
    target_rr=2.0,
    zone_approach_pct=0.3,
    zone_target_pct=0.5,
    min_risk_pct=0.005
)

if result:
    print(f"Current Price: ${current_price:.2f}")
    print(f"Resistance Zone: ${resistance_zone['low']:.2f} - ${resistance_zone['high']:.2f}")
    print(f"")
    print(f"✅ Trade Setup:")
    print(f"  Entry:       ${result['entry']:.2f}")
    print(f"  Stop Loss:   ${result['stop']:.2f}")
    print(f"  Take Profit: ${result['tp']:.2f}")
    print(f"  Risk:        ${result['risk']:.2f} ({(result['risk']/result['entry']*100):.2f}%)")
    print(f"  Reward:      ${result['reward']:.2f} ({(result['reward']/result['entry']*100):.2f}%)")
    print(f"  RR Ratio:    {result['rr_ratio']:.2f}:1")
    print(f"  Target Zone: {result['target_zone']}")
else:
    print("❌ No valid trade setup")

# Test Case 2: Bearish Trade
print("\n\n📉 TEST 2: BEARISH TRADE")
print("-" * 70)
current_price = 97.0  # Close to support zone (within 5%)
support_zone = {'low': 93.0, 'high': 95.0}
resistance_zone = {'low': 110.0, 'high': 112.0}

result = calculate_trade_levels_v4(
    current_price=current_price,
    bias='bear',
    support_zone=support_zone,
    resistance_zone=resistance_zone,
    target_rr=2.0,
    zone_approach_pct=0.3,
    zone_target_pct=0.5,
    min_risk_pct=0.005
)

if result:
    print(f"Current Price: ${current_price:.2f}")
    print(f"Support Zone: ${support_zone['low']:.2f} - ${support_zone['high']:.2f}")
    print(f"")
    print(f"✅ Trade Setup:")
    print(f"  Entry:       ${result['entry']:.2f}")
    print(f"  Stop Loss:   ${result['stop']:.2f}")
    print(f"  Take Profit: ${result['tp']:.2f}")
    print(f"  Risk:        ${result['risk']:.2f} ({(result['risk']/result['entry']*100):.2f}%)")
    print(f"  Reward:      ${result['reward']:.2f} ({(result['reward']/result['entry']*100):.2f}%)")
    print(f"  RR Ratio:    {result['rr_ratio']:.2f}:1")
    print(f"  Target Zone: {result['target_zone']}")
else:
    print("❌ No valid trade setup")

# Test Case 3: Invalid - Price too far from zone
print("\n\n❌ TEST 3: PRICE TOO FAR FROM TARGET ZONE (Should Reject)")
print("-" * 70)
current_price = 100.0
resistance_zone = {'low': 150.0, 'high': 152.0}  # Too far away

result = calculate_trade_levels_v4(
    current_price=current_price,
    bias='bull',
    support_zone=support_zone,
    resistance_zone=resistance_zone,
    target_rr=2.0,
    zone_approach_pct=0.3,
    zone_target_pct=0.5,
    min_risk_pct=0.005
)

if result:
    print("❌ ERROR: Trade should have been rejected!")
else:
    print("✅ CORRECT: Trade rejected (price too far from target zone)")

# Test Case 4: Invalid - Price already in target zone
print("\n\n❌ TEST 4: PRICE ALREADY IN TARGET ZONE (Should Reject)")
print("-" * 70)
current_price = 106.0  # Already in resistance zone
resistance_zone = {'low': 105.0, 'high': 107.0}

result = calculate_trade_levels_v4(
    current_price=current_price,
    bias='bull',
    support_zone=support_zone,
    resistance_zone=resistance_zone,
    target_rr=2.0,
    zone_approach_pct=0.3,
    zone_target_pct=0.5,
    min_risk_pct=0.005
)

if result:
    print("❌ ERROR: Trade should have been rejected!")
else:
    print("✅ CORRECT: Trade rejected (price already in target zone)")

print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)
