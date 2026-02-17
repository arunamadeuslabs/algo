"""
Configuration for Nifty Option Selling Backtest Strategy
EMA Crossover + Plus Sign System
"""

# --- EMA Settings ---
FAST_EMA_PERIOD = 9       # Orange MA
SLOW_EMA_PERIOD = 21      # Green MA

# --- Risk Management ---
STOP_LOSS_POINTS = 50          # Fixed SL in Nifty points
TARGET_POINTS = 100            # Fixed target (1:2 RR)
RISK_REWARD_RATIO = 2.0        # Minimum R:R
TRAILING_STOP_USE_EMA = True   # Trail SL using Green MA
MAX_LOSS_PER_DAY = 5000        # Max loss per day in INR

# --- Premium-based SL/Target (for option selling) ---
USE_PREMIUM_BASED_EXIT = False # Disabled - using spot-based exits (1:2 RR)
SL_PREMIUM_PCT = 0.30          # SL when premium rises 30% above entry
TARGET_PREMIUM_PCT = 0.60      # Target when premium decays 60% (1:2)

# --- Option Selling Settings ---
OPTION_LOT_SIZE = 25           # Nifty lot size (updated)
NUM_LOTS = 1                   # Number of lots to trade
STRIKE_SELECTION = "ATM"       # ATM / OTM1 / OTM2
OTM_OFFSET = 50                # OTM strike offset (for OTM selection)
PREMIUM_COLLECTION_MIN = 80    # Min premium to sell (INR)
DAYS_TO_EXPIRY_MAX = 7         # Max DTE for selling

# --- Filters ---
VOLUME_MULTIPLIER = 0.0        # 0 = disabled, 1.2 = signal candle volume must be 1.2x avg
SIDEWAYS_ADX_THRESHOLD = 10    # ADX below this = sideways (relaxed from 15)
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# --- Timeframe ---
TIMEFRAME = "5min"             # 1min / 5min / 15min / 1H / 4H / Daily
TRADING_START = "09:20"        # IST - skip first 5 min
TRADING_END = "15:15"          # IST - no new trades after this
SQUARE_OFF_TIME = "15:25"      # IST - forced square off

# --- Capital ---
INITIAL_CAPITAL = 500000       # 5 Lakh starting capital
MARGIN_PER_LOT = 100000        # Approx margin for 1 lot option selling

# --- Brokerage & Transaction Costs (per order) ---
BROKERAGE_PER_ORDER = 20       # Flat ₹20/order (Zerodha/Dhan style)
SLIPPAGE_POINTS = 2.0          # ₹2 slippage per option trade (bid-ask + delay)
STT_SELL_RATE = 0.000625       # STT on sell side: 0.0625% of (premium × qty)
EXCHANGE_TXN_RATE = 0.0005    # NSE transaction charge: 0.05%
SEBI_CHARGE_RATE = 0.000001    # SEBI charges: ₹10 per crore
STAMP_DUTY_BUY_RATE = 0.00003  # Stamp duty on buy side: 0.003%
GST_RATE = 0.18               # 18% GST on (brokerage + exchange charges)
INCLUDE_COSTS = True           # Set False to run without costs for comparison

# --- Data ---
NIFTY_SYMBOL = "NIFTY 50"
DATA_DIR = "data"
RESULTS_DIR = "results"
