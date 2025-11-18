"""
Configuration module for v4 backtest strategy.
Single Responsibility: Store all user-configurable parameters.

V4 CHANGES:
- Fixed stop loss based on support/resistance zones
- Target zones instead of fixed lines
- Risk-reward ratio determines stop loss placement
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
    "BCH-USD", "MATIC-USD", "UNI-USD"
]
INTERVAL = "15m"      # Options: "5m", "15m", "30m", "1h", "1d"
LIMIT = 100000       # Max bars to use for backtest
PERIOD = "60d"       # Options: "7d", "60d", "720d"

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
# STRATEGY PARAMETERS (V4)
# -------------------------
# Risk-Reward Ratio
TARGET_RR_RATIO = 3.0      # Risk-reward ratio (e.g., 2:1 means target is 2x the risk)

# Support/Resistance Zone Detection
SR_LOOKBACK = 30           # Bars to look back for S/R detection
SR_MIN_TOUCHES = 1         # Minimum touches to confirm S/R zone
SR_ZONE_TOLERANCE = 0.001  # Zone thickness as % of price (0.1% - very tight zones)

# Zone Entry/Exit Parameters
ZONE_APPROACH_PCT = 0.3    # How close to zone edge before considering "approaching" (30%)
ZONE_TARGET_PCT = 0.5      # Where in target zone to place TP (50% = middle of zone)

# Bias scoring thresholds
BIAS_BULL_THRESHOLD = 8.0
BIAS_BEAR_THRESHOLD = -8.0

# Minimum risk validation
MIN_RISK_PCT = 0.1         # Minimum risk as % of price (0.1% - very permissive)

# -------------------------
# BACKTEST SETTINGS
# -------------------------
MAX_TRADE_DURATION = 512   # Maximum trade duration in bars
                           # If in profit: close at this duration
                           # If in loss: close as well

# -------------------------
# PERFORMANCE SETTINGS
# -------------------------
MAX_WORKERS = min(6, mp.cpu_count())  # Limit concurrent processing
