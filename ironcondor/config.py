"""
Configuration for Bank Nifty Iron Condor Strategy
===================================================
Sell OTM Call Spread + OTM Put Spread on Bank Nifty weekly options.
High-probability credit spread with defined risk.
"""

# --- Capital & Position Sizing ---
INITIAL_CAPITAL = 400000         # ₹4 Lakh
OPTION_LOT_SIZE = 15             # Bank Nifty lot size
NUM_LOTS = 2                     # 2 lots per side

# --- Entry Settings ---
ENTRY_TIME = "10:05"             # Enter after 10 AM (let market settle)
ENTRY_LATEST_TIME = "12:00"      # Don't enter after noon
SKIP_EXPIRY_DAY = True           # Skip entry on Thursday (expiry day)

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
MAX_LOSS_PER_DAY = 15000         # Max loss per day in INR (~3.75% of capital)
DAILY_PROFIT_TARGET = 5000       # ₹5,000 daily target

# --- Range-Bound Filters ---
# Only enter when market is range-bound
VIX_MAX = 25.0                   # Don't enter if India VIX > 25
RSI_LOWER = 30                   # RSI must be between 30-70 for range-bound
RSI_UPPER = 70
RSI_PERIOD = 14
BB_PERIOD = 20                   # Bollinger Band period
BB_STD = 2.0                     # Bollinger Band standard deviations
BB_WIDTH_MAX = 0.08              # Max BB width (% of price) — range filter

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
BACKTEST_DAYS = 365              # 1 year backtest

# --- Premium Estimation ---
BANKNIFTY_IV = 16.0              # Approx Bank Nifty IV (%) — baseline
RISK_FREE_RATE = 6.5             # Risk free rate (%)
