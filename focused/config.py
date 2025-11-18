"""
Configuration module for backtest strategy.
Single Responsibility: Store all user-configurable parameters.
"""
import os
import multiprocessing as mp

# Get directory paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------
# TRADING SYMBOLS & DATA
# -------------------------
SYMBOLS = [
    "BTC-USD", "ETH-USD", "DOGE-USD", "SOL-USD", 
    "ADA-USD", "XRP-USD", "BNB-USD", "TRX-USD", "LINK-USD"
]
INTERVAL = "5m"      # Options: "5m", "15m", "30m", "1h", "1d"
LIMIT = 100000       # Max bars to use for backtest

# -------------------------
# CAPITAL & RISK MANAGEMENT
# -------------------------
INITIAL_CAPITAL = 20.0
LEVERAGE = 2
MAX_MARGIN = 100.0  # USD - Maximum margin per trade

# -------------------------
# FEE & SLIPPAGE
# -------------------------
TAKER_FEE_RATE = 0.0004  # 0.04%
SLIPPAGE_PCT = 0.0002    # 0.02%

# -------------------------
# STRATEGY PARAMETERS
# -------------------------
RR_FOLLOW = 2.5         # Risk-reward ratio
ATR_SL_MULT = 2.0       # ATR multiplier for stop loss

# Bias scoring thresholds
BIAS_BULL_THRESHOLD = 8.0
BIAS_BEAR_THRESHOLD = -8.0

# -------------------------
# BACKTEST SETTINGS
# -------------------------
ALLOW_OVERLAP = True    # Allow overlapping trades
MAX_BARS_AHEAD = 500    # Maximum bars to check for stop hits

# -------------------------
# PERFORMANCE SETTINGS
# -------------------------
MAX_WORKERS = min(4, mp.cpu_count())  # Limit concurrent processing
