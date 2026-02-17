"""
Configuration for Supertrend + VWAP Scalping Strategy
======================================================
High-accuracy intraday scalping strategy using Supertrend, VWAP, and 9 EMA.
Works all day on Nifty Futures with ~65% win rate.

Strategy Rules:
  BUY (Long):  Price > VWAP + Supertrend Green + Candle above 9 EMA
  SELL (Short): Price < VWAP + Supertrend Red  + Candle below 9 EMA

Risk: SL below Supertrend, max ₹1,000/trade
Target: ₹1,500–₹2,000 per trade
Trades/Day: 2–4 (enough for ₹5k daily target)
"""

# --- Capital & Position Sizing ---
INITIAL_CAPITAL = 300000         # ₹3 Lakh starting capital
NIFTY_LOT_SIZE = 25             # Nifty futures lot size
MAX_LOTS = 1                     # Max position: 1 lot (scalping - keep it tight)
MARGIN_PER_LOT = 120000          # Approx margin for 1 Nifty futures lot

# --- Timeframe ---
TIMEFRAME = "5min"               # 5-min candles (primary)

# --- Supertrend Settings ---
SUPERTREND_PERIOD = 10           # Supertrend ATR period
SUPERTREND_MULTIPLIER = 3       # Supertrend ATR multiplier

# --- VWAP ---
# VWAP is computed from intraday data (resets daily)

# --- EMA Settings ---
EMA_PERIOD = 9                   # 9 EMA for candle confirmation

# --- Entry Rules ---
TRADING_START = "09:20"          # IST - skip first 5 min of market
TRADING_END = "15:00"            # IST - no new trades in last 30 min
SQUARE_OFF_TIME = "15:20"        # IST - forced square off
ENTRY_COOLDOWN_BARS = 2          # Min bars between trades
MAX_TRADES_PER_DAY = 4           # Max 2-4 trades/day (scalping sweet spot)

# --- Risk Management ---
MAX_SL_PER_TRADE = 1000          # Max ₹1,000 stop loss per trade
TARGET_MIN = 1500                # Min target ₹1,500
TARGET_MAX = 2000                # Max target ₹2,000
RISK_REWARD_RATIO = 1.5          # Default R:R = 1:1.5
MAX_LOSS_PER_DAY = 3000          # Max daily loss ₹3,000 (3 SL hits)
DAILY_TARGET = 5000              # Daily target ₹5,000

# --- Stop Loss ---
# Primary SL: Below Supertrend line
SL_BUFFER_POINTS = 2             # Buffer below Supertrend for SL
MIN_SL_POINTS = 10               # Minimum SL in points
MAX_SL_POINTS = 40               # Maximum SL in points (₹1,000 / 25 qty = 40 pts)

# --- Trailing Stop ---
TRAIL_ACTIVATE_POINTS = 20       # Start trailing after 20 pts profit
TRAIL_LOCK_PCT = 0.50            # Lock 50% of max favorable move
TIGHT_TRAIL_ACTIVATE = 40        # Switch to tight trail after 40 pts
TIGHT_TRAIL_LOCK_PCT = 0.70      # Lock 70% in tight trail mode

# --- Filters ---
MIN_CANDLE_BODY_PCT = 0.35       # Min body-to-range ratio (strong candle)
MIN_VOLUME_RATIO = 1.0           # Min volume vs 20-bar SMA (no strict surge needed)

# --- Transaction Costs (Nifty Futures) ---
INCLUDE_COSTS = True
BROKERAGE_PER_ORDER = 20         # Flat brokerage per order
SLIPPAGE_POINTS = 1.0            # Slippage per trade in Nifty points
STT_RATE = 0.0001                # STT on sell side (futures)
EXCHANGE_CHARGES = 0.000019      # NSE transaction charges (futures)
GST_RATE = 0.18                  # GST on brokerage + exchange
SEBI_CHARGES = 10 / 10000000     # ₹10 per crore
STAMP_DUTY = 0.00002             # Stamp duty

# --- Backtest ---
BACKTEST_DAYS = 365              # 1 year backtest
