"""
Configuration for Nifty Sapphire Intraday Strategy
====================================================
Paired theta-focused short strangle with dynamic trailing SL.
Sell OTM Call + OTM Put at market open, trail both legs, exit EOD.
"""

# --- Capital & Position Sizing ---
INITIAL_CAPITAL = 250000         # ₹2.5 Lakh (part of ₹10L total)
OPTION_LOT_SIZE = 25             # Nifty lot size
NUM_LOTS = 2                     # Lots per leg

# --- Entry Settings ---
ENTRY_TIME = "09:20"             # Enter at market open
OTM_OFFSET_CE = 150              # Call leg: Nifty + 150 points OTM
OTM_OFFSET_PE = 150              # Put leg: Nifty - 150 points OTM
STRIKE_ROUNDING = 50             # Nifty strikes in multiples of 50
DAYS_TO_EXPIRY = 3               # Avg DTE for weekly — enters mid-week (max theta)

# --- Exit Settings ---
SQUARE_OFF_TIME = "15:25"        # EOD square-off
TRADING_START = "09:20"
TRADING_END = "15:15"

# --- Dynamic Trailing Stop Loss ---
# Phase 1: Initial SL (before any profit)
INITIAL_SL_PCT = 0.30            # 30% premium rise = SL hit (per leg)

# Phase 2: Lock-in trailing (after premium decays)
TRAIL_ACTIVATE_PCT = 0.15        # Start trailing after 15% premium decay
TRAIL_STEP_PCT = 0.08            # Trail in 8% steps
TRAIL_LOCK_PCT = 0.45            # Lock 45% of max favorable move

# Phase 3: Aggressive trailing (deep profit)
DEEP_PROFIT_PCT = 0.50           # If premium decays 50%+, use tight trail
DEEP_TRAIL_LOCK_PCT = 0.70       # Lock 70% of profits when deep

# --- Combined Position Management ---
COMBINED_SL_PCT = 0.25           # If combined premium rises 25% from entry, exit both
MAX_LOSS_PER_DAY = 12500         # Max loss per day in INR (5% of capital)

# --- Momentum Shift Adjustment ---
MOMENTUM_THRESHOLD = 100         # Nifty moves 100+ pts from entry = momentum
MOMENTUM_SHIFT_ENABLED = True    # Roll profitable leg closer on momentum
MOMENTUM_ROLL_OFFSET = 100       # Roll profitable leg to OTM-100 on momentum
RE_ENTRY_ENABLED = False         # Don't re-enter after exit (1 trade/day)

# --- Transaction Costs (Indian Market) ---
INCLUDE_COSTS = True
BROKERAGE_PER_ORDER = 20         # Flat brokerage per order
SLIPPAGE_POINTS = 2              # Slippage per leg in premium points
STT_RATE = 0.000625              # STT on sell side
EXCHANGE_CHARGES = 0.0005        # NSE transaction charges
GST_RATE = 0.18                  # GST on brokerage + exchange
SEBI_CHARGES = 10 / 10000000    # ₹10 per crore
STAMP_DUTY = 0.00003             # Stamp duty

# --- Data Settings ---
TIMEFRAME = "5min"               # 5-min candles for intraday
BACKTEST_DAYS = 90               # 3 months backtest

# --- Premium Estimation ---
NIFTY_IV = 12.5                  # Approx Nifty IV (%) — baseline, varies 10-25%
RISK_FREE_RATE = 6.5             # Risk free rate (%)
