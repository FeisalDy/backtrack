# Quick test with 30m data
import subprocess
import sys

symbols_30m = ["BTC", "ETH", "DOGE"]

print("Testing 30m data for all available symbols...\n")

for symbol in symbols_30m:
    print(f"\n{'='*60}")
    print(f"Testing {symbol} with 30m data")
    print('='*60)
    
    # Create a temporary test script
    with open('bias_v3.py', 'r') as f:
        content = f.read()
    
    # Modify symbol and interval
    content = content.replace('SYMBOL = "BTC"', f'SYMBOL = "{symbol}"')
    content = content.replace('INTERVAL = "hour"', 'INTERVAL = "30m"')
    
    with open('temp_test.py', 'w') as f:
        f.write(content)
    
    # Run the test
    result = subprocess.run([sys.executable, 'temp_test.py'], capture_output=True, text=True)
    
    # Print only the results section
    if "===== BACKTEST RESULTS =====" in result.stdout:
        print(result.stdout.split("===== BACKTEST RESULTS =====")[1].split("Results saved")[0])

import os
os.remove('temp_test.py')
print("\nAll tests complete!")
