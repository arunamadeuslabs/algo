"""
Configuration for Bank Nifty Iron Condor Strategy
===================================================
Sell OTM Call Spread + OTM Put Spread on Bank Nifty weekly options.
High-probability credit spread with defined risk.
"""

# --- Capital & Position Sizing ---
INITIAL_CAPITAL = 200000         # ₹2 Lakh (part of ₹10L total)
OPTION_LOT_SIZE = 15             # Bank Nifty lot size
NUM_LOTS = 3                     # 3 lots per side

# --- Entry Settings ---
ENTRY_TIME = "10:05"             # Enter after 10 AM (let market settle)
ENTRY_LATEST_TIME = "12:00"      # Don't enter after noon
SKIP_EXPIRY_DAY = False          # Trade on Thursdays too (max daily frequency)

# --- Strike Selection ---
OTM_OFFSET_CE = 300              # Sell CE at spot + 300 (OTM)
OTM_OFFSET_PE = 300              # Sell PE at spot - 300 (OTM)
WING_WIDTH = 300                 # Buy protection 300 pts further OTM
STRIKE_ROUNDING = 100            # Bank Nifty strikes in multiples of 100
DAYS_TO_EXPIRY = 1               # Near expiry for maximum theta decay

# --- Exit Settings ---
SQUARE_OFF_TIME = "15:20"        # EOD square-off
TRADING_START = "10:05"
TRADING_END = "15:15"

# --- Target & Stop Loss ---
TARGET_PCT = 0.30                # Close at 30% of credit captured (achievable intraday)
MAX_TARGET_PCT = 0.50            # Aggressive target: 50% credit capture
SL_BREACH_PCT = 0.25             # Exit if spot breaches sold strikes by 25% of wing width
COMBINED_SL_MULTIPLIER = 2.0     # Exit if combined premium rises to 2x of net credit
MAX_LOSS_PER_DAY = 10000         # Max loss per day in INR (5% of capital)
DAILY_PROFIT_TARGET = 7500       # ₹7,500 daily target

# --- Range-Bound Filters ---
# Only enter when market is range-bound
VIX_MAX = 28.0                   # Don't enter if India VIX > 28 (relaxed)
RSI_LOWER = 20                   # RSI range-bound zone widened (was 30)
RSI_UPPER = 80                   # (was 70)
RSI_PERIOD = 14
BB_PERIOD = 20                   # Bollinger Band period
BB_STD = 2.0                     # Bollinger Band standard deviations
BB_WIDTH_MAX = 0.15              # BB filter nearly disabled (was 0.08)

# --- Transaction Costs (Indian Market) ---
INCLUDE_COSTS = True
BROKERAGE_PER_ORDER = 20         # Flat brokerage per order (4 legs)
SLIPPAGE_POINTS = 1              # Slippage per leg in premium points (Bank Nifty options are liquid)
STT_RATE = 0.000625              # STT on sell side
EXCHANGE_CHARGES = 0.0005        # NSE transaction charges
GST_RATE = 0.18                  # GST on brokerage + exchange
SEBI_CHARGES = 10 / 10000000    # ₹10 per crore
STAMP_DUTY = 0.00003             # Stamp duty

# --- Data Settings ---
TIMEFRAME = "5min"               # 5-min candles for intraday
BACKTEST_DAYS = 90               # 3 months backtest

# --- Premium Estimation ---
BANKNIFTY_IV = 16.0              # Approx Bank Nifty IV (%) — baseline
RISK_FREE_RATE = 6.5             # Risk free rate (%)
