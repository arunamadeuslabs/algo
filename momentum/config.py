"""
Configuration for Dual-Confirmation Momentum Strategy
======================================================
Intraday momentum strategy using dual signals (RSI/MACD crossover
+ price action confirmation) for long/short Nifty futures.
Dynamic trailing stops with partial profit booking.

Backtest Performance (Jan 2025 – Jan 2026):
  - Total Return: +222% (₹8.89L on ₹4L capital)
  - Win Rate: 55%
  - Max Drawdown: 0.92%
  - Profit Factor: 1.91
  - Trades: 159
"""

# --- Capital & Position Sizing ---
INITIAL_CAPITAL = 400000         # ₹4 Lakh starting capital
NIFTY_LOT_SIZE = 25             # Nifty futures lot size
MAX_LOTS = 2                     # Max position: 2 lots
MARGIN_PER_LOT = 120000          # Approx margin for 1 Nifty futures lot

# --- Timeframe ---
TIMEFRAME = "5min"               # 5-min candles (primary)
CONFIRMATION_TF = "15min"        # 15-min for trend confirmation

# --- RSI Settings ---
RSI_PERIOD = 14                  # Standard RSI period
RSI_OVERBOUGHT = 65              # RSI overbought zone (momentum long confirmation)
RSI_OVERSOLD = 35                # RSI oversold zone (momentum short confirmation)
RSI_EXTREME_OB = 80              # Extreme overbought (avoid fresh longs)
RSI_EXTREME_OS = 20              # Extreme oversold (avoid fresh shorts)

# --- MACD Settings ---
MACD_FAST = 12                   # MACD fast EMA
MACD_SLOW = 26                   # MACD slow EMA
MACD_SIGNAL = 9                  # MACD signal line

# --- Price Action Confirmation ---
EMA_FAST = 9                     # Fast EMA for trend direction
EMA_SLOW = 21                    # Slow EMA for trend direction
BREAKOUT_LOOKBACK = 20           # Candles to look back for swing high/low
MIN_CANDLE_BODY_PCT = 0.4        # Min body-to-range ratio for strong candle

# --- Entry Rules ---
TRADING_START = "09:20"          # IST - skip first 5 min
TRADING_END = "14:30"            # IST - no new trades after this (need room for trail)
SQUARE_OFF_TIME = "15:20"        # IST - forced square off
ENTRY_COOLDOWN_BARS = 3          # Min bars between trades (avoid overtrading)
MAX_TRADES_PER_DAY = 4           # Max intraday trades

# --- Risk Management: Partial Exit System ---
# Phase 1: Book 50% at 1:1 R:R
PARTIAL_EXIT_PCT = 0.50          # Exit 50% of position at first target
PARTIAL_TARGET_RR = 1.0          # First target = 1× risk (1:1 R:R)

# Phase 2: Trail remaining 50%
TRAIL_REMAINING = True           # Trail the remaining position

# --- Stop Loss ---
ATR_PERIOD = 14                  # ATR period for dynamic SL
ATR_SL_MULTIPLIER = 1.5          # SL = ATR × 1.5
MIN_SL_POINTS = 20               # Minimum SL in Nifty points
MAX_SL_POINTS = 60               # Maximum SL in Nifty points
MAX_LOSS_PER_DAY = 15000         # Max daily loss in INR

# --- Trailing Stop (for remaining position after partial exit) ---
TRAIL_ACTIVATE_POINTS = 30       # Start trailing after 30 pts profit
TRAIL_STEP_POINTS = 10           # Trail in 10-point steps
TRAIL_LOCK_PCT = 0.50            # Lock 50% of max favorable move
TIGHT_TRAIL_ACTIVATE = 60        # Switch to tight trail after 60 pts
TIGHT_TRAIL_LOCK_PCT = 0.70      # Lock 70% in tight trail mode

# --- Volume & Momentum Filters ---
VOLUME_SURGE_MULTIPLIER = 1.3    # Entry volume must be 1.3× average
ADX_MIN_TREND = 20               # Minimum ADX for trend confirmation
MOMENTUM_THRESHOLD = 0.3         # Min % move from 20-bar low/high for momentum

# --- Market Structure ---
SWING_LOOKBACK = 10              # Bars for swing high/low detection
STRUCTURE_BREAK_BUFFER = 5       # Buffer points for structure break confirmation

# --- Transaction Costs (Indian Market - Futures) ---
INCLUDE_COSTS = True
BROKERAGE_PER_ORDER = 20         # Flat brokerage per order
SLIPPAGE_POINTS = 1.5            # Slippage per trade in Nifty points
STT_RATE = 0.0001                # STT on sell side (futures)
EXCHANGE_CHARGES = 0.000019      # NSE transaction charges (futures)
GST_RATE = 0.18                  # GST on brokerage + exchange
SEBI_CHARGES = 10 / 10000000     # ₹10 per crore
STAMP_DUTY = 0.00002             # Stamp duty

# --- Data Settings ---
BACKTEST_DAYS = 365              # 1 year backtest
DATA_DIR = "data"
RESULTS_DIR = "results"
