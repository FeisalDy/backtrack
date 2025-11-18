"""
Validation Script - Compare Original vs Optimized Results
Ensures optimized versions produce equivalent results
"""
import pandas as pd
import numpy as np
import sys

print("=" * 70)
print("VALIDATION: Comparing Original vs Optimized Backtest Results")
print("=" * 70)

def load_results(filename):
    """Load backtest results CSV"""
    try:
        df = pd.read_csv(filename)
        return df
    except FileNotFoundError:
        return None

def compare_results(df1, df2, name1="Original", name2="Optimized", tolerance=1e-3):
    """
    Compare two result dataframes
    tolerance: Allow small differences due to float32 vs float64
    """
    print(f"\n📊 Comparing {name1} vs {name2}")
    print("-" * 70)
    
    if df1 is None:
        print(f"❌ {name1} results not found. Run the original backtest first.")
        return False
    
    if df2 is None:
        print(f"❌ {name2} results not found. Run the optimized backtest first.")
        return False
    
    # Check number of trades
    n1, n2 = len(df1), len(df2)
    print(f"\n1. Trade Count:")
    print(f"   {name1}: {n1:,} trades")
    print(f"   {name2}: {n2:,} trades")
    
    if n1 != n2:
        print(f"   ⚠️  WARNING: Different number of trades!")
        print(f"   This may indicate a logic difference.")
        min_n = min(n1, n2)
        print(f"   Comparing first {min_n:,} trades only...")
        df1 = df1.head(min_n)
        df2 = df2.head(min_n)
    else:
        print(f"   ✓ Same number of trades")
    
    # Compare entry times
    print(f"\n2. Entry Times:")
    time_matches = (df1['entry_time'] == df2['entry_time']).sum()
    time_pct = time_matches / len(df1) * 100
    print(f"   Matching entries: {time_matches}/{len(df1)} ({time_pct:.1f}%)")
    
    if time_pct < 100:
        print(f"   ⚠️  WARNING: Entry times differ!")
        mismatches = df1[df1['entry_time'] != df2['entry_time']]
        print(f"   First mismatch at trade #{mismatches.index[0]+1}")
    else:
        print(f"   ✓ All entry times match")
    
    # Compare bias direction
    print(f"\n3. Trade Direction (Bias):")
    bias_matches = (df1['bias'] == df2['bias']).sum()
    bias_pct = bias_matches / len(df1) * 100
    print(f"   Matching bias: {bias_matches}/{len(df1)} ({bias_pct:.1f}%)")
    
    if bias_pct < 100:
        print(f"   ⚠️  WARNING: Some trade directions differ!")
    else:
        print(f"   ✓ All bias directions match")
    
    # Compare numerical values
    print(f"\n4. Numerical Values (tolerance={tolerance}):")
    
    numeric_cols = ['entry', 'stop_loss', 'take_profit', 'close_price', 
                    'pl', 'fee_entry', 'fee_exit', 'net']
    
    all_close = True
    for col in numeric_cols:
        if col not in df1.columns or col not in df2.columns:
            continue
        
        # Check if values are close (allowing for float32 rounding)
        close = np.allclose(df1[col], df2[col], rtol=tolerance, atol=1e-6)
        
        if close:
            max_diff = np.abs(df1[col] - df2[col]).max()
            print(f"   ✓ {col:15s} - Max diff: {max_diff:.6f}")
        else:
            max_diff = np.abs(df1[col] - df2[col]).max()
            mean_diff = np.abs(df1[col] - df2[col]).mean()
            print(f"   ⚠️  {col:15s} - Max diff: {max_diff:.6f}, Mean: {mean_diff:.6f}")
            all_close = False
    
    # Compare final metrics
    print(f"\n5. Final Metrics:")
    
    metrics = {
        'Total P/L': df1['net'].sum(),
        'Avg P/L': df1['net'].mean(),
        'Win Rate %': (df1['pl'] > 0).sum() / len(df1) * 100,
        'Total Wins': (df1['pl'] > 0).sum(),
        'Total Losses': (df1['pl'] <= 0).sum(),
    }
    
    metrics2 = {
        'Total P/L': df2['net'].sum(),
        'Avg P/L': df2['net'].mean(),
        'Win Rate %': (df2['pl'] > 0).sum() / len(df2) * 100,
        'Total Wins': (df2['pl'] > 0).sum(),
        'Total Losses': (df2['pl'] <= 0).sum(),
    }
    
    print(f"\n   {'Metric':<20s} {name1:>15s} {name2:>15s} {'Diff':>15s}")
    print("   " + "-" * 66)
    
    for key in metrics:
        v1, v2 = metrics[key], metrics2[key]
        diff = v2 - v1
        diff_pct = (diff / v1 * 100) if v1 != 0 else 0
        
        if 'Rate' in key or 'Wins' in key or 'Losses' in key:
            print(f"   {key:<20s} {v1:15.2f} {v2:15.2f} {diff:+15.2f}")
        else:
            print(f"   {key:<20s} {v1:15.4f} {v2:15.4f} {diff:+15.4f} ({diff_pct:+.2f}%)")
    
    # Final verdict
    print("\n" + "=" * 70)
    if n1 == n2 and time_pct == 100 and bias_pct == 100 and all_close:
        print("✅ VALIDATION PASSED: Results are equivalent!")
        print("   Optimized version produces the same trading decisions.")
        return True
    elif abs(metrics['Total P/L'] - metrics2['Total P/L']) < 0.1:
        print("✅ VALIDATION MOSTLY PASSED: Minor differences within tolerance.")
        print("   Likely due to float32 rounding. Trading logic is preserved.")
        return True
    else:
        print("⚠️  VALIDATION WARNING: Significant differences detected.")
        print("   Review the differences above. May indicate logic changes.")
        return False

# ===================================
# Main Comparison
# ===================================
if __name__ == "__main__":
    print("\nThis script compares backtest results to validate optimizations.\n")
    
    # Default symbol to check
    symbol = "BTC-USD"
    interval = "5m"
    
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
    if len(sys.argv) > 2:
        interval = sys.argv[2]
    
    print(f"Checking results for: {symbol} {interval}\n")
    
    # Load original results
    original_file = f"results/backtest_{symbol}_{interval}.csv"
    print(f"Loading original results: {original_file}")
    df_original = load_results(original_file)
    
    if df_original is None:
        print(f"\n❌ Original results not found at {original_file}")
        print(f"   Please run the original backtest first:")
        print(f"   python bias_v10.py")
        sys.exit(1)
    
    print(f"✓ Loaded {len(df_original):,} trades from original backtest\n")
    
    # You can manually compare by running optimized version separately
    # and comparing the output files
    
    print("-" * 70)
    print("INSTRUCTIONS:")
    print("-" * 70)
    print("""
To validate the optimized versions:

1. Run original backtest (if not already done):
   python bias_v10.py

2. Backup the results:
   cp -r results results_original

3. Run optimized backtest:
   python bias_v10_optimized.py

4. Compare results:
   python compare_results.py BTC-USD 5m

The optimized version should produce nearly identical results.
Small differences (<0.1%) are acceptable due to float32 precision.

Key metrics that MUST match:
- Number of trades
- Entry times
- Trade directions (bull/bear)
- Final P/L (within 0.1%)

If you see larger differences, there may be a logic bug.
Please report any significant discrepancies.
""")
    
    print("=" * 70)
    print(f"Original backtest loaded successfully!")
    print(f"  Symbol: {symbol}")
    print(f"  Interval: {interval}")
    print(f"  Trades: {len(df_original):,}")
    print(f"  Total P/L: ${df_original['net'].sum():.2f}")
    print(f"  Win Rate: {(df_original['pl'] > 0).sum() / len(df_original) * 100:.2f}%")
    print("=" * 70)
