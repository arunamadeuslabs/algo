"""
Configuration for Supertrend + VWAP Scalping Strategy (Optimized v2)
=====================================================================
High-accuracy intraday scalping strategy using Supertrend, VWAP, 9 EMA,
ADX trend filter, and VWAP distance filter.

Strategy Rules:
  BUY (Long):  Price > VWAP + Supertrend Green + Candle above 9 EMA
               + ADX > 20 (trending) + Price 10+ pts from VWAP
  SELL (Short): Price < VWAP + Supertrend Red  + Candle below 9 EMA
               + ADX > 20 (trending) + Price 10+ pts from VWAP

Risk: SL below Supertrend, max ₹750/trade, R:R 1:2
Target: ₹1,500–₹2,500 per trade
Trades/Day: 2–3 (fewer, higher-quality trades)
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
SUPERTREND_MULTIPLIER = 3.0     # Standard multiplier

# --- VWAP ---
# VWAP is computed from intraday data (resets daily)

# --- EMA Settings ---
EMA_PERIOD = 9                   # 9 EMA for candle confirmation

# --- ADX Trend Filter ---
ADX_PERIOD = 14                  # ADX lookback period
ADX_THRESHOLD = 25               # Min ADX for entry (strong trend only)

# --- VWAP Distance Filter ---
MIN_VWAP_DISTANCE = 8            # Min pts away from VWAP (avoid noise zone)

# --- Supertrend Stability ---
SUPERTREND_CONFIRM_BARS = 2      # Supertrend must be same direction for N bars

# --- Entry Rules ---
TRADING_START = "09:20"          # IST - skip first 5 min of market
TRADING_END = "15:00"            # IST - no new trades in last 30 min
SQUARE_OFF_TIME = "15:20"        # IST - forced square off
ENTRY_COOLDOWN_BARS = 4          # Min bars between trades
MAX_TRADES_PER_DAY = 2           # Max 2 trades/day (quality over quantity)

# --- Risk Management ---
MAX_SL_PER_TRADE = 1000          # Max ₹1,000 stop loss per trade
TARGET_MIN = 1500                # Min target ₹1,500
TARGET_MAX = 2500                # Max target ₹2,500 (was ₹2,000 — let winners run)
RISK_REWARD_RATIO = 2.0          # Default R:R = 1:2 (was 1:1.5)
MAX_LOSS_PER_DAY = 2500          # Max daily loss ₹2,500
DAILY_TARGET = 5000              # Daily target ₹5,000

# --- Stop Loss ---
# Primary SL: Below Supertrend line
SL_BUFFER_POINTS = 2             # Buffer below Supertrend for SL
MIN_SL_POINTS = 10               # Minimum SL in points
MAX_SL_POINTS = 40               # Maximum SL in points

# --- Trailing Stop (Breakeven + Supertrend Trail) ---
BREAKEVEN_ACTIVATE_POINTS = 20   # Move SL to breakeven after 20 pts favorable
BREAKEVEN_LOCK_POINTS = 8        # Lock 8 pts profit at breakeven (covers costs)
TRAIL_WITH_SUPERTREND = True     # Trail SL using Supertrend line (natural trend trail)
TRAIL_ACTIVATE_POINTS = 30       # Legacy: start pct trailing after 30 pts (fallback)
TRAIL_LOCK_PCT = 0.35            # Legacy: lock 35% of max favorable
TIGHT_TRAIL_ACTIVATE = 55        # Legacy: tight trail after 55 pts
TIGHT_TRAIL_LOCK_PCT = 0.55      # Legacy: lock 55% in tight trail

# --- Filters ---
MIN_CANDLE_BODY_PCT = 0.40       # Min body-to-range ratio (was 0.35 — stronger candles only)
MIN_VOLUME_RATIO = 1.0           # Min volume vs 20-bar SMA (no strict surge needed)
REQUIRE_STRONG_CANDLE = True     # Require strong candle confirmation (no weak entries)

# --- Transaction Costs (Nifty Futures) ---
INCLUDE_COSTS = True
BROKERAGE_PER_ORDER = 20         # Flat brokerage per order
SLIPPAGE_POINTS = 0.5            # Slippage per trade in Nifty points (was 1.0)
STT_RATE = 0.0001                # STT on sell side (futures)
EXCHANGE_CHARGES = 0.000019      # NSE transaction charges (futures)
GST_RATE = 0.18                  # GST on brokerage + exchange
SEBI_CHARGES = 10 / 10000000     # ₹10 per crore
STAMP_DUTY = 0.00002             # Stamp duty

# --- Backtest ---
BACKTEST_DAYS = 365              # 1 year backtest
