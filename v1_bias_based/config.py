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
    "ADA-USD", "XRP-USD", "BNB-USD", "TRX-USD", 
    "LINK-USD", "AVAX-USD", "DOT-USD", "LTC-USD", 
    "BCH-USD", "MATIC-USD", "UNI-USD", "LTC-USD", 
    "BCH-USD"
]
INTERVAL = "5m"      # Options: "5m", "15m", "30m", "1h", "1d"
LIMIT = 100000       # Max bars to use for backtest
PERIOD = "60d" #7d, 60d, 720d

# -------------------------
# CAPITAL & RISK MANAGEMENT
# -------------------------
INITIAL_CAPITAL = 20.0
LEVERAGE = 2
MAX_MARGIN = 60.0  # USD - Maximum margin per trade

# -------------------------
# FEE & SLIPPAGE
# -------------------------
TAKER_FEE_RATE = 0.0004  # 0.04%
SLIPPAGE_PCT = 0.0002    # 0.02%

# -------------------------
# STRATEGY PARAMETERS
# -------------------------
RR_FOLLOW = 2         # Risk-reward ratio (minimum required)
MIN_RISK_PCT = 0.5      # Minimum risk as % of price (0.5% = realistic S/R zones)
ATR_SL_MULT = 2.0       # ATR multiplier for stop loss (legacy, not used in new S/R strategy)
ZONE_PENETRATION = 0.2  # How deep into S/R zone to enter (20%)
SR_LOOKBACK = 30        # Bars to look back for S/R detection
SR_MIN_TOUCHES = 1      # Minimum touches to confirm S/R zone

# Bias scoring thresholds
BIAS_BULL_THRESHOLD = 9.0
BIAS_BEAR_THRESHOLD = -9.0

# -------------------------
# BACKTEST SETTINGS
# -------------------------
MAX_BARS_AHEAD = 64    # Maximum bars to check for stop hits
MAX_TRADE_DURATION = 512  # Maximum trade duration in bars (8 hours = 32 bars for 15m interval)
                         # If in profit: close at this duration
                         # If in loss: keep open for another cycle

# -------------------------
# PERFORMANCE SETTINGS
# -------------------------
MAX_WORKERS = min(6, mp.cpu_count())  # Limit concurrent processing
