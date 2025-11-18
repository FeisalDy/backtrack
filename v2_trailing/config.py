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
# STRATEGY PARAMETERS - V2 TRAILING STOP
# -------------------------
RR_FOLLOW = 2.5              # Initial risk-reward ratio for TP
MIN_RISK_PCT = 0.05          # Minimum risk as % of price (0.05% = very loose)

# Swing Detection (for stop loss placement)
SWING_LOOKBACK = 30          # Bars to look back for swing high/low (more bars = more swings found)
SWING_TOLERANCE_PCT = 0.5    # Tolerance zone around swing (0.5% = wide area)

# Pullback Entry (not strictly enforced anymore)
PULLBACK_TO_ZONE_PCT = 0.2   # Not used - kept for compatibility
ZONE_DEPTH_PCT = 0.2         # S/R zone depth as % of price

# Trailing Stop
TRAIL_ACTIVATION_R = 1.0     # Start trailing after 1R profit (1x risk)
TRAIL_DISTANCE_PCT = 0.5     # Trail stop at 0.5% below high (bull) or above low (bear)
BREAKEVEN_AT_R = 0.5         # Move stop to breakeven after 0.5R profit

# Support/Resistance
SR_LOOKBACK = 100            # Bars to look back for S/R detection (large lookback for 1h)
SR_MIN_TOUCHES = 2           # Minimum touches to confirm S/R zone

# Bias scoring thresholds
BIAS_BULL_THRESHOLD = 3.0    # Lowered from 5.0 - very lenient
BIAS_BEAR_THRESHOLD = -3.0   # Lowered from -5.0 - very lenient

# -------------------------
# BACKTEST SETTINGS
# -------------------------
MAX_BARS_AHEAD = 1000        # Maximum bars to check for stop hits (no time limit, let price decide)

# -------------------------
# PERFORMANCE SETTINGS
# -------------------------
MAX_WORKERS = min(6, mp.cpu_count())  # Limit concurrent processing
