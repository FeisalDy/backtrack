"""
Configuration module for v5 backtest strategy.
Single Responsibility: Store all user-configurable parameters.

V5 STRATEGY:
- Uses bias_v8 BEST_RESULT logic for bias scoring
- ATR-based stop loss (not S/R zones)
- Enhanced bias scoring with multiple indicators
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
    "ADA-USD", "XRP-USD", "BNB-USD", "TRX-USD", 
    "LINK-USD", "AVAX-USD", "DOT-USD", "LTC-USD", 
    "BCH-USD"
]
INTERVAL = "15m"     # Options: "5m", "15m", "30m", "1h", "1d"
LIMIT = 100000       # Max bars to use for backtest
PERIOD = "max"       # Options: "7d", "60d", "720d"

# -------------------------
# CAPITAL & RISK MANAGEMENT
# -------------------------
INITIAL_CAPITAL = 20.0
LEVERAGE = 10
# Position sizing: Always trade 3% of current capital (implemented in backtest_engine.py)

# -------------------------
# FEE & SLIPPAGE
# -------------------------
TAKER_FEE_RATE = 0.0004  # 0.04%
SLIPPAGE_PCT = 0.0002    # 0.02%

# -------------------------
# STRATEGY PARAMETERS (V5 - BIAS V8)
# -------------------------
# Risk-Reward and Stop Loss
RR_FOLLOW = 3        # Risk-reward ratio
ATR_SL_MULT = 2.0      # ATR multiplier for stop loss

# Bias scoring thresholds (from bias_v8)
BIAS_BULL_THRESHOLD = 8.0
BIAS_BEAR_THRESHOLD = -8.0

# -------------------------
# BACKTEST SETTINGS
# -------------------------
MAX_TRADE_DURATION = 512   # Maximum trade duration in bars
ALLOW_OVERLAP = False      # Don't allow overlapping trades

# -------------------------
# PERFORMANCE SETTINGS
# -------------------------
MAX_WORKERS = min(6, mp.cpu_count())  # Limit concurrent processing
