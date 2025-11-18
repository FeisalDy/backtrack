"""
Visualization module.
Single Responsibility: Generate all charts and visualizations for backtest results.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for parallel processing
import matplotlib.pyplot as plt
import os
from config import RESULTS_DIR, INITIAL_CAPITAL, INTERVAL


def plot_equity_curve(equity_curve, symbol, initial_capital, output_dir=None):
    """
    Plot equity curve with profit/loss fill and drawdown.
    
    Args:
        equity_curve: List of equity values over time
        symbol: Trading symbol
        initial_capital: Starting capital
        output_dir: Directory to save the chart (defaults to RESULTS_DIR)
    """
    if output_dir is None:
        output_dir = RESULTS_DIR
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    equity = np.array(equity_curve)
    trades = np.arange(len(equity))
    
    # Equity curve
    ax1.plot(trades, equity, linewidth=2, color='#2E86AB', label='Equity')
    ax1.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    ax1.fill_between(trades, initial_capital, equity, where=(equity >= initial_capital), 
                      alpha=0.3, color='green', label='Profit')
    ax1.fill_between(trades, initial_capital, equity, where=(equity < initial_capital), 
                      alpha=0.3, color='red', label='Loss')
    ax1.set_ylabel('Equity ($)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{symbol} - Equity Curve', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Drawdown
    running_max = np.maximum.accumulate(equity)
    drawdown = (running_max - equity) / running_max * 100
    ax2.fill_between(trades, 0, drawdown, color='red', alpha=0.3)
    ax2.plot(trades, drawdown, color='darkred', linewidth=1.5)
    ax2.set_xlabel('Trade Number', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{INTERVAL}_{symbol}_equity_curve.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_trade_distribution(res_df, symbol, output_dir=None):
    """
    Plot P&L distribution, cumulative P&L, wins/losses, and exit types.
    
    Args:
        res_df: DataFrame with trade results
        symbol: Trading symbol
        output_dir: Directory to save the chart (defaults to RESULTS_DIR)
    """
    if output_dir is None:
        output_dir = RESULTS_DIR
    
    if res_df.empty:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # P&L Distribution
    ax1 = axes[0, 0]
    net_pls = res_df['net'].values
    ax1.hist(net_pls, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
    ax1.axvline(x=np.mean(net_pls), color='green', linestyle='--', linewidth=2, 
                label=f'Mean: ${np.mean(net_pls):.2f}')
    ax1.set_xlabel('Net P&L ($)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Trade P&L Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Cumulative P&L
    ax2 = axes[0, 1]
    cumulative_pl = np.cumsum(net_pls)
    ax2.plot(cumulative_pl, linewidth=2, color='#A23B72')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.fill_between(range(len(cumulative_pl)), 0, cumulative_pl, 
                      where=(cumulative_pl >= 0), alpha=0.3, color='green')
    ax2.fill_between(range(len(cumulative_pl)), 0, cumulative_pl, 
                      where=(cumulative_pl < 0), alpha=0.3, color='red')
    ax2.set_xlabel('Trade Number', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Cumulative P&L ($)', fontsize=11, fontweight='bold')
    ax2.set_title('Cumulative P&L', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Win/Loss by Bias
    ax3 = axes[1, 0]
    bias_results = res_df.groupby('bias').apply(
        lambda x: pd.Series({
            'wins': (x['net'] > 0).sum(),
            'losses': (x['net'] <= 0).sum()
        })
    )
    bias_results.plot(kind='bar', ax=ax3, color=['green', 'red'], alpha=0.7)
    ax3.set_xlabel('Bias', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax3.set_title('Wins vs Losses by Bias', fontsize=12, fontweight='bold')
    ax3.legend(['Wins', 'Losses'])
    ax3.grid(True, alpha=0.3, axis='y')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0)
    
    # Exit Type Distribution
    ax4 = axes[1, 1]
    exit_counts = res_df['exit_type'].value_counts()
    colors_exit = {'tp': 'green', 'sl': 'red', 'mtm': 'orange', 'time_profit': 'blue'}
    colors = [colors_exit.get(x, 'gray') for x in exit_counts.index]
    ax4.pie(exit_counts.values, labels=exit_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    ax4.set_title('Exit Type Distribution', fontsize=12, fontweight='bold')
    
    plt.suptitle(f'{symbol} - Trade Analysis', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{INTERVAL}_{symbol}_trade_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_performance_comparison(all_results):
    """
    Create comparison charts across all tested symbols.
    
    Args:
        all_results: List of result dictionaries from all symbols
    """
    if not all_results:
        return
    
    df = pd.DataFrame(all_results)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Net P&L comparison
    ax1 = axes[0, 0]
    net_pls = [float(x.replace('$', '')) for x in df['Net P/L']]
    colors = ['green' if x > 0 else 'red' for x in net_pls]
    ax1.barh(df['Symbol'], net_pls, color=colors, alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax1.set_xlabel('Net P&L ($)', fontsize=11, fontweight='bold')
    ax1.set_title('Net P&L by Symbol', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Win Rate comparison
    ax2 = axes[0, 1]
    win_rates = [float(x) for x in df['Win Rate %']]
    ax2.barh(df['Symbol'], win_rates, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(x=50, color='orange', linestyle='--', linewidth=2, label='50% Threshold')
    ax2.set_xlabel('Win Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Win Rate by Symbol', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Max Drawdown comparison
    ax3 = axes[1, 0]
    max_dds = [float(x) for x in df['Max DD %']]
    ax3.barh(df['Symbol'], max_dds, color='darkred', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Max Drawdown (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Maximum Drawdown by Symbol', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Number of Trades
    ax4 = axes[1, 1]
    ax4.barh(df['Symbol'], df['Trades'], color='purple', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Number of Trades', fontsize=11, fontweight='bold')
    ax4.set_title('Trade Count by Symbol', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle(f'Performance Comparison - {INTERVAL} Timeframe', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{INTERVAL}_performance_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
