"""
Main orchestrator for backtesting strategy.
Single Responsibility: Coordinate all modules and execute the complete backtest workflow.

This is the entry point for running backtests across multiple symbols in parallel.
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from config import SYMBOLS, INTERVAL, LIMIT, MAX_WORKERS, PERIOD
from backtest_engine import backtest_single_symbol
from visualization import plot_performance_comparison


def main():
    """
    Execute complete backtest workflow:
    1. Configure parameters (from config.py)
    2. Run backtests in parallel across all symbols
    3. Generate individual charts per symbol
    4. Create performance comparison across symbols
    5. Display summary results
    """
    print("=" * 70)
    print("BACKTEST STRATEGY - BIAS SCORING SYSTEM")
    print("=" * 70)
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print(f"Timeframe: {INTERVAL}")
    print(f"Max bars: {LIMIT:,}")
    print(f"Parallel workers: {MAX_WORKERS}")
    print(f"Results directory: ./results/")
    print("=" * 70)
    print()
    
    print(f"Starting backtest with {MAX_WORKERS} parallel workers...")
    
    # Prepare arguments for parallel processing
    args_list = [(symbol, INTERVAL, LIMIT, PERIOD) for symbol in SYMBOLS]
    
    # Execute backtests in parallel
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        all_results = list(executor.map(backtest_single_symbol, args_list))
    
    # Filter out failed backtests
    all_results = [r for r in all_results if r is not None]

    if all_results:
        # Display summary table
        summary_df = pd.DataFrame(all_results)
        print("\n\n" + "=" * 70)
        print("SUMMARY BACKTEST RESULTS")
        print("=" * 70)
        print(summary_df.to_string(index=False))
        print("=" * 70)
        
        # Generate performance comparison chart
        print("\n\nGenerating performance comparison chart...")
        plot_performance_comparison(all_results)
        
        print("\n✅ Backtest completed successfully!")
        print(f"📊 All results saved to: ./results/")
    else:
        print("\n❌ No successful backtests completed.")


if __name__ == "__main__":
    main()
