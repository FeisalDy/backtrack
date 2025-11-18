"""
Debug test to understand the validation logic.
"""
from support_resistance import calculate_trade_levels_v4

# Test Case 1: Bullish Trade
print("=" * 70)
print("DEBUG: BULLISH TRADE")
print("=" * 70)
current_price = 103.0
support_zone = {'low': 90.0, 'high': 92.0}
resistance_zone = {'low': 105.0, 'high': 107.0}

print(f"Current Price: ${current_price:.2f}")
print(f"Support Zone: ${support_zone['low']:.2f} - ${support_zone['high']:.2f}")
print(f"Resistance Zone: ${resistance_zone['low']:.2f} - ${resistance_zone['high']:.2f}")
print(f"")
print(f"Distance to resistance: ${resistance_zone['low'] - current_price:.2f}")
print(f"10% of current price: ${current_price * 0.10:.2f}")
print(f"Price < res_low? {current_price < resistance_zone['low']}")

result = calculate_trade_levels_v4(
    current_price=current_price,
    bias='bull',
    support_zone=support_zone,
    resistance_zone=resistance_zone,
    target_rr=2.0,
    min_risk_pct=0.005
)

if result:
    print(f"\n✅ Trade Setup Generated:")
    print(f"  Entry: ${result['entry']:.2f}")
    print(f"  TP: ${result['tp']:.2f}")
    print(f"  Stop: ${result['stop']:.2f}")
    print(f"  Risk: ${result['risk']:.2f} ({(result['risk']/result['entry']*100):.2f}%)")
    print(f"  Reward: ${result['reward']:.2f}")
    
    # Check if stop is above support
    if result['stop'] > support_zone['high']:
        print(f"\n⚠️  Stop ${result['stop']:.2f} is ABOVE support zone high ${support_zone['high']:.2f}")
else:
    print("\n❌ No trade setup generated")
