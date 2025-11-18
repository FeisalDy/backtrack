"""
Debug test with detailed step-by-step validation.
"""

# Manual calculation
current_price = 103.0
res_low = 105.0
res_high = 107.0
sup_low = 90.0
sup_high = 92.0

print("=" * 70)
print("STEP-BY-STEP CALCULATION")
print("=" * 70)

# Step 1: Calculate TP
zone_size = res_high - res_low
tp = res_low + (zone_size * 0.5)
print(f"1. Zone size: ${zone_size:.2f}")
print(f"2. TP (50% into zone): ${tp:.2f}")

# Step 2: Calculate reward
reward = tp - current_price
print(f"3. Reward: ${reward:.2f}")

# Step 3: Calculate risk from RR ratio
target_rr = 2.0
risk = reward / target_rr
print(f"4. Risk (reward / RR): ${risk:.2f}")

# Step 4: Calculate stop
stop = current_price - risk
print(f"5. Stop Loss: ${stop:.2f}")

# Step 5: Check if stop conflicts with support
print(f"\n6. Support zone: ${sup_low:.2f} - ${sup_high:.2f}")
print(f"7. Stop > sup_high? {stop} > {sup_high} = {stop > sup_high}")

if stop > sup_high:
    print(f"\n❌ REJECTED: Stop loss ${stop:.2f} would be ABOVE support zone high ${sup_high:.2f}")
    print(f"   This creates a conflict - stop should not be in/above support")
else:
    print(f"\n✅ VALID: Stop loss ${stop:.2f} is below support zone")

# Step 6: Check minimum risk %
risk_pct = (risk / current_price) * 100
print(f"\n8. Risk %: {risk_pct:.2f}%")
print(f"9. Min risk %: 0.50%")
if risk_pct < 0.5:
    print("❌ REJECTED: Risk too small")
else:
    print("✅ VALID: Risk is adequate")
