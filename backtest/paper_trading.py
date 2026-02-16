"""
Paper Trading Engine for Nifty Option Selling Strategy
=======================================================
Runs LIVE during market hours, fetches real-time candles from Dhan API,
applies the EMA Crossover strategy, and simulates trades without placing
real orders.

Features:
  - Live candle polling from Dhan API (configurable interval)
  - Same strategy logic as backtest (EMA crossover + filters)
  - Virtual position tracking with P&L
  - Trade log persistence (CSV)
  - Console dashboard with live status
  - Auto square-off at 15:25 IST
  - Daily summary at market close

Usage:
  python paper_trading.py                     # Default 5min candles
  python paper_trading.py --interval 15       # 15min candles
  python paper_trading.py --capital 200000    # Custom capital
  python paper_trading.py --resume            # Resume from saved state
"""

import sys
import os
import json
import time
import signal as sig
import argparse
import logging
from datetime import datetime, timedelta, date
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List

import pandas as pd
import numpy as np

# Add backtest directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from data_utils import compute_indicators, get_atm_strike, estimate_option_premium, generate_sample_nifty_data
from dhan_fetch import fetch_nifty_intraday, fetch_nifty_ltp, fetch_nifty_ohlc
from strategy import (
    Signal, TradeStatus, Trade, calculate_transaction_costs,
)

# ── Constants ────────────────────────────────────────────────
PAPER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_trades")
STATE_FILE = os.path.join(PAPER_DIR, "state.json")
TRADE_LOG_CSV = os.path.join(PAPER_DIR, "paper_trade_log.csv")
DAILY_LOG_CSV = os.path.join(PAPER_DIR, "daily_summary.csv")

# How many historical candles to bootstrap indicators (need >= slow EMA period + buffer)
WARMUP_CANDLES = 60

# Polling interval padding (seconds after candle close to wait for data)
POLL_DELAY_SEC = 10

# ── Logging Setup ────────────────────────────────────────────
os.makedirs(PAPER_DIR, exist_ok=True)

logger = logging.getLogger("PaperTrader")
logger.setLevel(logging.DEBUG)

# Console handler
_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter("%(asctime)s │ %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(_ch)

# File handler
_fh = logging.FileHandler(os.path.join(PAPER_DIR, "paper_trading.log"), encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter("%(asctime)s │ %(levelname)-7s │ %(message)s"))
logger.addHandler(_fh)


# ── Paper Trade Dataclass ────────────────────────────────────
@dataclass
class PaperTrade:
    """A simulated trade with full tracking."""
    id: int
    entry_time: str
    direction: str           # "BUY" (sell PUT) or "SELL" (sell CALL)
    option_type: str         # "CE" or "PE"
    strike: float
    entry_spot: float
    entry_premium: float
    sl_premium: float
    target_premium: float
    sl_spot: float
    target_spot: float
    lots: int = NUM_LOTS
    status: str = "OPEN"      # OPEN / TARGET_HIT / SL_HIT / TIME_EXIT / FORCED_EXIT
    exit_time: str = ""
    exit_spot: float = 0.0
    exit_premium: float = 0.0
    gross_pnl: float = 0.0
    costs: float = 0.0
    net_pnl: float = 0.0
    max_favorable: float = 0.0

    @property
    def quantity(self):
        return self.lots * OPTION_LOT_SIZE

    def to_dict(self):
        d = asdict(self)
        d["quantity"] = self.quantity
        return d


# ── Paper Trading Engine ─────────────────────────────────────
class PaperTradingEngine:
    """
    Live paper trading engine using the EMA Crossover option selling strategy.
    Polls Dhan API for new candles and applies the same strategy logic.
    """

    def __init__(self, interval: int = 5, capital: float = INITIAL_CAPITAL,
                 fast_ema: int = FAST_EMA_PERIOD, slow_ema: int = SLOW_EMA_PERIOD,
                 resume: bool = False):
        self.interval = interval            # Candle interval in minutes
        self.initial_capital = capital
        self.capital = capital
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema

        self.current_trade: Optional[PaperTrade] = None
        self.closed_trades: List[PaperTrade] = []
        self.trade_counter = 0
        self.daily_loss = 0.0
        self.today = date.today()
        self.pending_signal: Optional[Signal] = None

        # Candle data buffer
        self.data: Optional[pd.DataFrame] = None
        self.last_bar_time: Optional[pd.Timestamp] = None

        self._running = False

        # Signal diagnostics (daily)
        self._reset_diag()

        if resume:
            self._load_state()

    # ── Signal Generation (same logic as backtest) ───────────
    def _generate_signal(self, row: pd.Series, bar_time) -> Signal:
        """Generate signal from latest indicator row."""
        is_bull = row.get("crossover_bull", False)
        is_bear = row.get("crossover_bear", False)

        if not (is_bull or is_bear):
            return Signal.NONE

        self.diag["total_crossovers"] += 1

        # Filter 1: Close vs MAs
        close = row["close"]
        ema_slow = row["ema_slow"]
        if is_bull and close <= ema_slow:
            self.diag["filtered_close_vs_ma"] += 1
            return Signal.NONE
        if is_bear and close >= ema_slow:
            self.diag["filtered_close_vs_ma"] += 1
            return Signal.NONE

        # Filter 2: Volume
        if not row.get("high_volume", False):
            self.diag["filtered_volume"] += 1
            return Signal.NONE

        # Filter 3: ADX (not sideways)
        if not row.get("not_sideways", True):
            self.diag["filtered_adx"] += 1
            return Signal.NONE

        # Filter 4: Trading hours
        if hasattr(bar_time, 'hour'):
            time_str = f"{bar_time.hour:02d}:{bar_time.minute:02d}"
            if time_str < TRADING_START or time_str > TRADING_END:
                self.diag["filtered_time"] += 1
                return Signal.NONE

        self.diag["signals_passed"] += 1

        if is_bull:
            return Signal.BUY
        elif is_bear:
            return Signal.SELL
        return Signal.NONE

    # ── Trade Entry ──────────────────────────────────────────
    def _enter_trade(self, row: pd.Series, bar_time, signal: Signal):
        """Enter a paper trade."""
        spot = row["close"]

        if signal == Signal.BUY:
            option_type = "PE"
            strike = get_atm_strike(spot)
            sl_spot = min(row.get("recent_swing_low", spot - STOP_LOSS_POINTS),
                          row["ema_slow"]) - 10
            target_spot = spot + TARGET_POINTS
        else:
            option_type = "CE"
            strike = get_atm_strike(spot)
            sl_spot = max(row.get("recent_swing_high", spot + STOP_LOSS_POINTS),
                          row["ema_slow"]) + 10
            target_spot = spot - TARGET_POINTS

        dte = DAYS_TO_EXPIRY_MAX
        entry_premium = estimate_option_premium(spot, strike, dte, option_type)

        if entry_premium < PREMIUM_COLLECTION_MIN * 0.5:
            self.diag["premium_too_low"] += 1
            logger.info(f"  ⚠️  Premium too low ({entry_premium:.2f}), skipping")
            return

        # Premium-based SL/Target
        if USE_PREMIUM_BASED_EXIT:
            sl_premium = entry_premium * (1 + SL_PREMIUM_PCT)
            target_premium = max(5.0, entry_premium * (1 - SL_PREMIUM_PCT * RISK_REWARD_RATIO))
        else:
            sl_premium = 0.0
            target_premium = 0.0

        self.trade_counter += 1
        trade = PaperTrade(
            id=self.trade_counter,
            entry_time=str(bar_time),
            direction="BUY" if signal == Signal.BUY else "SELL",
            option_type=option_type,
            strike=strike,
            entry_spot=spot,
            entry_premium=entry_premium,
            sl_premium=sl_premium,
            target_premium=target_premium,
            sl_spot=sl_spot,
            target_spot=target_spot,
            lots=NUM_LOTS,
        )
        self.current_trade = trade
        self.diag["trades_entered"] += 1

        logger.info("=" * 55)
        logger.info(f"  📝 PAPER TRADE ENTERED  #{trade.id}")
        logger.info(f"     Direction : {'BULLISH → Sell PUT' if signal == Signal.BUY else 'BEARISH → Sell CALL'}")
        logger.info(f"     Option    : {option_type} {strike}")
        logger.info(f"     Spot      : {spot:.2f}")
        logger.info(f"     Premium   : ₹{entry_premium:.2f}")
        logger.info(f"     SL Prem   : ₹{sl_premium:.2f}  |  Target Prem: ₹{target_premium:.2f}")
        logger.info(f"     Qty       : {trade.quantity} ({NUM_LOTS} lot)")
        logger.info("=" * 55)

        self._save_state()

    # ── Trade Management ─────────────────────────────────────
    def _manage_trade(self, row: pd.Series, bar_time):
        """Check SL, target, trailing stop, time exit for open trade."""
        if self.current_trade is None:
            return

        trade = self.current_trade
        spot = row["close"]

        # Estimate current premium
        entry_dt = pd.Timestamp(trade.entry_time)
        bars_elapsed = (bar_time - entry_dt).total_seconds() / 3600
        dte_remaining = max(0.1, DAYS_TO_EXPIRY_MAX - bars_elapsed / 6.5)
        current_premium = estimate_option_premium(
            spot, trade.strike, int(dte_remaining), trade.option_type
        )

        # Track max favorable
        premium_decay = trade.entry_premium - current_premium
        trade.max_favorable = max(trade.max_favorable, premium_decay)

        # Premium-based exits
        if USE_PREMIUM_BASED_EXIT:
            if current_premium >= trade.sl_premium:
                self._exit_trade(bar_time, spot, current_premium, "SL_HIT")
                return
            if current_premium <= trade.target_premium:
                self._exit_trade(bar_time, spot, current_premium, "TARGET_HIT")
                return
        else:
            # Spot-based exits
            if trade.direction == "BUY":
                if row["low"] <= trade.sl_spot:
                    self._exit_trade(bar_time, spot, current_premium, "SL_HIT")
                    return
                if spot >= trade.target_spot:
                    self._exit_trade(bar_time, spot, current_premium, "TARGET_HIT")
                    return
            else:
                if row["high"] >= trade.sl_spot:
                    self._exit_trade(bar_time, spot, current_premium, "SL_HIT")
                    return
                if spot <= trade.target_spot:
                    self._exit_trade(bar_time, spot, current_premium, "TARGET_HIT")
                    return

        # Time-based exit
        if hasattr(bar_time, 'hour'):
            time_str = f"{bar_time.hour:02d}:{bar_time.minute:02d}"
            if time_str >= SQUARE_OFF_TIME:
                self._exit_trade(bar_time, spot, current_premium, "TIME_EXIT")
                return

        # Max daily loss
        unrealized = (trade.entry_premium - current_premium) * trade.quantity
        if self.daily_loss + min(0, unrealized) < -MAX_LOSS_PER_DAY:
            self._exit_trade(bar_time, spot, current_premium, "FORCED_EXIT")
            return

        # Trailing Stop using Slow EMA (only for spot-based mode)
        if TRAILING_STOP_USE_EMA and not USE_PREMIUM_BASED_EXIT:
            ema_slow = row.get("ema_slow", 0)
            if ema_slow > 0:
                if trade.direction == "BUY":  # Sold PUT
                    new_trailing = max(trade.sl_spot, ema_slow - 10)
                    trade.sl_spot = new_trailing
                else:  # Sold CALL
                    new_trailing = min(trade.sl_spot, ema_slow + 10)
                    if trade.sl_spot == 0 or new_trailing < trade.sl_spot:
                        trade.sl_spot = new_trailing

        # Print live P&L
        logger.debug(f"  📊 Open P&L: ₹{unrealized:+,.2f}  |  Premium: {current_premium:.2f}  |  Spot: {spot:.2f}")

    # ── Trade Exit ───────────────────────────────────────────
    def _exit_trade(self, bar_time, spot: float, exit_premium: float, status: str):
        """Close paper trade and record results."""
        trade = self.current_trade
        if trade is None:
            return

        trade.exit_time = str(bar_time)
        trade.exit_spot = spot
        trade.exit_premium = exit_premium
        trade.status = status

        # Calculate P&L
        trade.gross_pnl = (trade.entry_premium - exit_premium) * trade.quantity
        costs = calculate_transaction_costs(trade.entry_premium, exit_premium, trade.quantity)
        trade.costs = costs["total"]
        trade.net_pnl = trade.gross_pnl - trade.costs

        self.capital += trade.net_pnl
        self.daily_loss += min(0, trade.net_pnl)
        self.closed_trades.append(trade)
        self.current_trade = None

        # Status emoji
        emoji = {"TARGET_HIT": "🎯", "SL_HIT": "🛑", "TIME_EXIT": "⏰", "FORCED_EXIT": "⚠️"}
        status_icon = emoji.get(status, "❓")

        logger.info("=" * 55)
        logger.info(f"  {status_icon} TRADE CLOSED  #{trade.id}  [{status}]")
        logger.info(f"     {trade.option_type} {trade.strike}")
        logger.info(f"     Entry Prem : ₹{trade.entry_premium:.2f}  →  Exit: ₹{exit_premium:.2f}")
        logger.info(f"     Gross P&L  : ₹{trade.gross_pnl:+,.2f}")
        logger.info(f"     Costs      : ₹{trade.costs:,.2f}")
        logger.info(f"     Net P&L    : ₹{trade.net_pnl:+,.2f}")
        logger.info(f"     Capital    : ₹{self.capital:,.2f}")
        logger.info("=" * 55)

        self._append_trade_csv(trade)
        self._save_state()

    # ── Data Fetching ────────────────────────────────────────
    def _fetch_latest_candles(self) -> pd.DataFrame:
        """
        Fetch recent candles from Dhan API and compute indicators.
        We fetch enough history for EMA warmup + current session.
        """
        days_back = max(5, (WARMUP_CANDLES * self.interval) // (375) + 3)
        days_back = min(days_back, 90)

        df = fetch_nifty_intraday(interval=self.interval, days_back=days_back)

        if df.empty:
            return pd.DataFrame()

        df = compute_indicators(df, fast_period=self.fast_ema, slow_period=self.slow_ema)
        return df

    def _get_new_bars(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return only bars newer than the last processed bar."""
        if self.last_bar_time is None:
            return df.tail(1)  # First run: just process latest bar
        mask = df.index > self.last_bar_time
        return df[mask]

    # ── Main Loop ────────────────────────────────────────────
    def run(self):
        """
        Main paper trading loop.
        Polls for new candles and processes them through the strategy.
        """
        self._running = True

        # Register graceful shutdown
        def _shutdown(signum, frame):
            logger.info("\n\n  🛑 Shutting down paper trader...")
            self._running = False
        sig.signal(sig.SIGINT, _shutdown)
        sig.signal(sig.SIGTERM, _shutdown)

        self._print_banner()

        while self._running:
            now = datetime.now()

            # Reset daily trackers at start of new day
            if now.date() != self.today:
                self._daily_summary()
                self.today = now.date()
                self.daily_loss = 0.0
                self._reset_diag()

            # Check if market is open (Mon-Fri, 9:15 - 15:30 IST)
            if not self._is_market_open(now):
                if now.hour >= 15 and now.minute >= 35:
                    # After market close — print summary and wait for next day
                    if self.current_trade is not None:
                        logger.info("  ⏰ Market closed — force closing open trade")
                        last_spot = self.data.iloc[-1]["close"] if self.data is not None and len(self.data) > 0 else 0
                        last_prem = estimate_option_premium(
                            last_spot, self.current_trade.strike, 1, self.current_trade.option_type
                        ) if last_spot > 0 else self.current_trade.entry_premium
                        self._exit_trade(now, last_spot, last_prem, "TIME_EXIT")
                    self._daily_summary()
                    logger.info("  😴 Market closed. Waiting for next session...")
                    self._wait_until_premarket()
                    continue
                else:
                    self._print_waiting(now)
                    time.sleep(60)
                    continue

            # Fetch latest candles
            logger.debug(f"  📡 Fetching {self.interval}min candles...")
            df = self._fetch_latest_candles()

            if df.empty:
                logger.warning("  ⚠️  No data received, retrying in 30s...")
                time.sleep(30)
                continue

            self.data = df
            new_bars = self._get_new_bars(df)

            if new_bars.empty:
                # No new candle yet, wait
                sleep_sec = self.interval * 60 // 2  # Half the candle interval
                logger.debug(f"  ⏳ No new candle, sleeping {sleep_sec}s...")
                time.sleep(sleep_sec)
                continue

            # Process each new bar
            for bar_time, row in new_bars.iterrows():
                self.last_bar_time = bar_time
                logger.info(f"\n  🕐 {bar_time}  |  O:{row['open']:.2f}  H:{row['high']:.2f}  L:{row['low']:.2f}  C:{row['close']:.2f}  V:{row['volume']:.0f}")

                # Manage existing trade
                if self.current_trade is not None:
                    self._manage_trade(row, bar_time)
                    # Check if signal fires while in trade
                    s = self._generate_signal(row, bar_time)
                    if s != Signal.NONE:
                        self.diag["blocked_by_open_trade"] += 1
                else:
                    # Execute pending signal from previous bar
                    if self.pending_signal is not None:
                        self._enter_trade(row, bar_time, self.pending_signal)
                        self.pending_signal = None
                    else:
                        # Generate new signal
                        s = self._generate_signal(row, bar_time)
                        if s != Signal.NONE and self.current_trade is None:
                            self.pending_signal = s
                            logger.info(f"  ⚡ Signal: {'BULLISH (Sell PUT)' if s == Signal.BUY else 'BEARISH (Sell CALL)'}  — entering on NEXT candle")

                # Live dashboard
                self._print_status(row, bar_time)

            # Sleep until next candle
            sleep_sec = self.interval * 60 + POLL_DELAY_SEC
            logger.debug(f"  💤 Sleeping {sleep_sec}s until next candle...")
            time.sleep(sleep_sec)

        # Graceful shutdown
        if self.current_trade is not None:
            logger.info("  Closing open trade on shutdown...")
            now = datetime.now()
            if self.data is not None and len(self.data) > 0:
                last_row = self.data.iloc[-1]
                spot = last_row["close"]
                prem = estimate_option_premium(spot, self.current_trade.strike, 1,
                                               self.current_trade.option_type)
                self._exit_trade(now, spot, prem, "FORCED_EXIT")

        self._daily_summary()
        self._save_state()
        logger.info("  ✅ Paper trader stopped. State saved.")

    # ── Market Hours ─────────────────────────────────────────
    def _is_market_open(self, now: datetime) -> bool:
        """Check if market is currently open."""
        if now.weekday() >= 5:  # Saturday/Sunday
            return False
        t = now.hour * 100 + now.minute
        return 915 <= t <= 1530

    def _wait_until_premarket(self):
        """Sleep until 9:10 AM next trading day."""
        now = datetime.now()
        next_open = now.replace(hour=9, minute=10, second=0, microsecond=0)
        if now >= next_open:
            next_open += timedelta(days=1)
        # Skip weekends
        while next_open.weekday() >= 5:
            next_open += timedelta(days=1)

        wait_sec = (next_open - now).total_seconds()
        if wait_sec > 0:
            logger.info(f"  Next session: {next_open.strftime('%Y-%m-%d %H:%M')} ({wait_sec/3600:.1f}h)")
            # Sleep in chunks so ctrl+C works
            while wait_sec > 0 and self._running:
                chunk = min(wait_sec, 300)
                time.sleep(chunk)
                wait_sec -= chunk

    # ── Persistence ──────────────────────────────────────────
    def _save_state(self):
        """Save current state to disk for resume."""
        state = {
            "capital": self.capital,
            "initial_capital": self.initial_capital,
            "trade_counter": self.trade_counter,
            "daily_loss": self.daily_loss,
            "today": str(self.today),
            "last_bar_time": str(self.last_bar_time) if self.last_bar_time else None,
            "interval": self.interval,
            "current_trade": self.current_trade.to_dict() if self.current_trade else None,
            "pending_signal": self.pending_signal.value if self.pending_signal else None,
        }
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def _load_state(self):
        """Load saved state from disk."""
        if not os.path.exists(STATE_FILE):
            logger.info("  No saved state found. Starting fresh.")
            return

        try:
            with open(STATE_FILE, "r") as f:
                state = json.load(f)

            self.capital = state.get("capital", self.initial_capital)
            self.trade_counter = state.get("trade_counter", 0)
            self.daily_loss = state.get("daily_loss", 0.0)
            self.interval = state.get("interval", self.interval)

            saved_today = state.get("today")
            if saved_today and saved_today == str(date.today()):
                self.daily_loss = state.get("daily_loss", 0.0)
            else:
                self.daily_loss = 0.0

            lbt = state.get("last_bar_time")
            if lbt:
                self.last_bar_time = pd.Timestamp(lbt)

            ps = state.get("pending_signal")
            if ps is not None:
                self.pending_signal = Signal(ps)

            ct = state.get("current_trade")
            if ct:
                ct.pop("quantity", None)  # Remove computed property
                self.current_trade = PaperTrade(**ct)
                logger.info(f"  📂 Resumed open trade #{self.current_trade.id}: {self.current_trade.option_type} {self.current_trade.strike}")

            # Load closed trades from CSV
            if os.path.exists(TRADE_LOG_CSV):
                csv_df = pd.read_csv(TRADE_LOG_CSV)
                today_trades = csv_df[csv_df["entry_time"].str.startswith(str(date.today()))]
                logger.info(f"  📂 Found {len(csv_df)} historical trades ({len(today_trades)} today)")

            logger.info(f"  📂 State restored. Capital: ₹{self.capital:,.2f}")

        except Exception as e:
            logger.error(f"  ❌ Failed to load state: {e}. Starting fresh.")

    def _append_trade_csv(self, trade: PaperTrade):
        """Append a closed trade to the CSV log."""
        df = pd.DataFrame([trade.to_dict()])
        header = not os.path.exists(TRADE_LOG_CSV)
        df.to_csv(TRADE_LOG_CSV, mode="a", header=header, index=False)

    # ── Display ──────────────────────────────────────────────
    def _print_banner(self):
        logger.info("")
        logger.info("=" * 58)
        logger.info("  📄 NIFTY PAPER TRADING ENGINE")
        logger.info("  Strategy: EMA Crossover + Plus Sign System")
        logger.info(f"  EMA: {self.fast_ema}/{self.slow_ema}  |  Interval: {self.interval}min")
        logger.info(f"  Capital: ₹{self.capital:,.2f}  |  RR: 1:{RISK_REWARD_RATIO:.0f}")
        if USE_PREMIUM_BASED_EXIT:
            logger.info(f"  SL: {SL_PREMIUM_PCT*100:.0f}% premium rise  |  Target: {SL_PREMIUM_PCT*RISK_REWARD_RATIO*100:.0f}% decay")
        else:
            logger.info(f"  SL: {STOP_LOSS_POINTS} pts  |  Target: {TARGET_POINTS} pts  |  Trailing: EMA{SLOW_EMA_PERIOD}")
        logger.info(f"  Lot Size: {OPTION_LOT_SIZE} × {NUM_LOTS} lots")
        logger.info(f"  Max Daily Loss: ₹{MAX_LOSS_PER_DAY:,.0f}")
        logger.info(f"  Hours: {TRADING_START} - {TRADING_END}  |  Square-off: {SQUARE_OFF_TIME}")
        logger.info("=" * 58)
        logger.info("  Press Ctrl+C to stop gracefully\n")

    def _print_status(self, row: pd.Series, bar_time):
        """Print live dashboard line."""
        spot = row["close"]
        ema_f = row.get("ema_fast", 0)
        ema_s = row.get("ema_slow", 0)
        adx = row.get("adx", 0)
        trend = "↑" if ema_f > ema_s else "↓"

        status_parts = [
            f"  {trend} Spot: {spot:.2f}",
            f"EMA {self.fast_ema}: {ema_f:.2f}",
            f"EMA {self.slow_ema}: {ema_s:.2f}",
            f"ADX: {adx:.1f}",
        ]

        if self.current_trade:
            t = self.current_trade
            # Estimate unrealized P&L
            entry_dt = pd.Timestamp(t.entry_time)
            hrs = (bar_time - entry_dt).total_seconds() / 3600
            dte = max(0.1, DAYS_TO_EXPIRY_MAX - hrs / 6.5)
            cp = estimate_option_premium(spot, t.strike, int(dte), t.option_type)
            upnl = (t.entry_premium - cp) * t.quantity
            status_parts.append(f"│ OPEN: {t.option_type} {t.strike} P&L: ₹{upnl:+,.0f}")

        status_parts.append(f"│ Cap: ₹{self.capital:,.0f}")
        logger.info("  ".join(status_parts))

    def _print_waiting(self, now: datetime):
        """Print waiting-for-market message."""
        t = now.hour * 100 + now.minute
        if t < 915:
            logger.info(f"  ⏳ Pre-market... ({now.strftime('%H:%M')})")
        elif now.weekday() >= 5:
            logger.info(f"  📅 Weekend. Market opens Monday.")

    def _reset_diag(self):
        """Reset daily diagnostics."""
        self.diag = {
            "total_crossovers": 0,
            "filtered_close_vs_ma": 0,
            "filtered_volume": 0,
            "filtered_adx": 0,
            "filtered_time": 0,
            "signals_passed": 0,
            "blocked_by_open_trade": 0,
            "premium_too_low": 0,
            "trades_entered": 0,
        }

    def _daily_summary(self):
        """Print and save end-of-day summary."""
        # Get today's trades
        today_trades = [t for t in self.closed_trades
                        if t.entry_time.startswith(str(self.today))]

        if not today_trades and self.diag["total_crossovers"] == 0:
            return

        total_pnl = sum(t.net_pnl for t in today_trades)
        wins = [t for t in today_trades if t.net_pnl > 0]
        losses = [t for t in today_trades if t.net_pnl <= 0]

        logger.info("\n" + "=" * 58)
        logger.info(f"  📊 DAILY SUMMARY — {self.today}")
        logger.info("=" * 58)
        logger.info(f"  Trades         : {len(today_trades)}")
        logger.info(f"  Wins / Losses  : {len(wins)} / {len(losses)}")
        if today_trades:
            wr = len(wins) / len(today_trades) * 100
            logger.info(f"  Win Rate       : {wr:.1f}%")
        logger.info(f"  Day P&L        : ₹{total_pnl:+,.2f}")
        logger.info(f"  Capital        : ₹{self.capital:,.2f}")
        logger.info(f"  Total Return   : {(self.capital - self.initial_capital) / self.initial_capital * 100:+.2f}%")
        logger.info("")
        logger.info("  Signal Diagnostics:")
        logger.info(f"    Crossovers     : {self.diag['total_crossovers']}")
        logger.info(f"    Passed filters : {self.diag['signals_passed']}")
        logger.info(f"    Trades entered : {self.diag['trades_entered']}")
        logger.info("=" * 58)

        # Append to daily CSV
        summary = {
            "date": str(self.today),
            "trades": len(today_trades),
            "wins": len(wins),
            "losses": len(losses),
            "day_pnl": round(total_pnl, 2),
            "capital": round(self.capital, 2),
            "crossovers": self.diag["total_crossovers"],
            "signals_passed": self.diag["signals_passed"],
        }
        df = pd.DataFrame([summary])
        header = not os.path.exists(DAILY_LOG_CSV)
        df.to_csv(DAILY_LOG_CSV, mode="a", header=header, index=False)

    # ── One-shot simulation mode (for testing) ───────────────
    def run_once(self, data: pd.DataFrame = None):
        """
        Process all bars in a given DataFrame (for testing/simulation).
        Does not poll the API — just runs through the data in one shot.
        """
        if data is None:
            logger.info("  Fetching data for one-shot simulation...")
            data = self._fetch_latest_candles()
            if data.empty:
                logger.error("  ❌ No data available.")
                return

        self.data = data
        self._print_banner()
        logger.info(f"  📊 Simulation mode: {len(data)} bars")
        logger.info(f"     Range: {data.index[0]} → {data.index[-1]}\n")

        for bar_time, row in data.iterrows():
            # Reset daily tracker
            current_date = bar_time.date() if hasattr(bar_time, 'date') else bar_time
            if current_date != self.today:
                if self.today != date.today() or self.diag["trades_entered"] > 0:
                    self._daily_summary()
                self.today = current_date
                self.daily_loss = 0.0
                self._reset_diag()

            self.last_bar_time = bar_time

            # Manage existing trade
            if self.current_trade is not None:
                self._manage_trade(row, bar_time)
                s = self._generate_signal(row, bar_time)
                if s != Signal.NONE:
                    self.diag["blocked_by_open_trade"] += 1
            else:
                if self.pending_signal is not None:
                    self._enter_trade(row, bar_time, self.pending_signal)
                    self.pending_signal = None
                else:
                    s = self._generate_signal(row, bar_time)
                    if s != Signal.NONE and self.current_trade is None:
                        self.pending_signal = s
                        logger.info(f"  ⚡ Signal: {'BULLISH' if s == Signal.BUY else 'BEARISH'} — entering next candle")

        # Force close any open trade
        if self.current_trade is not None:
            last_row = data.iloc[-1]
            last_time = data.index[-1]
            spot = last_row["close"]
            prem = estimate_option_premium(spot, self.current_trade.strike, 1,
                                           self.current_trade.option_type)
            self._exit_trade(last_time, spot, prem, "TIME_EXIT")

        self._daily_summary()
        self._save_state()

        # Final stats
        total_pnl = sum(t.net_pnl for t in self.closed_trades)
        logger.info("\n" + "=" * 58)
        logger.info("  📋 PAPER TRADING SIMULATION COMPLETE")
        logger.info(f"  Total Trades  : {len(self.closed_trades)}")
        logger.info(f"  Total P&L     : ₹{total_pnl:+,.2f}")
        logger.info(f"  Final Capital : ₹{self.capital:,.2f}")
        logger.info(f"  Return        : {(self.capital - self.initial_capital) / self.initial_capital * 100:+.2f}%")
        logger.info(f"  Trade log     : {TRADE_LOG_CSV}")
        logger.info(f"  State file    : {STATE_FILE}")
        logger.info("=" * 58)


# ── CLI Entry Point ──────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Nifty Paper Trading — EMA Crossover Option Selling"
    )
    parser.add_argument("--interval", type=int, default=5, choices=[1, 5, 15],
                        help="Candle interval in minutes (default: 5)")
    parser.add_argument("--capital", type=float, default=INITIAL_CAPITAL,
                        help=f"Starting capital (default: {INITIAL_CAPITAL})")
    parser.add_argument("--fast-ema", type=int, default=FAST_EMA_PERIOD,
                        help=f"Fast EMA period (default: {FAST_EMA_PERIOD})")
    parser.add_argument("--slow-ema", type=int, default=SLOW_EMA_PERIOD,
                        help=f"Slow EMA period (default: {SLOW_EMA_PERIOD})")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from saved state")
    parser.add_argument("--simulate", action="store_true",
                        help="Run one-shot simulation on recent data (no live polling)")
    parser.add_argument("--live", action="store_true",
                        help="Run live paper trading with Dhan API data")
    parser.add_argument("--days", type=int, default=5,
                        help="Days of data for simulation mode (default: 5)")

    args = parser.parse_args()

    engine = PaperTradingEngine(
        interval=args.interval,
        capital=args.capital,
        fast_ema=args.fast_ema,
        slow_ema=args.slow_ema,
        resume=args.resume,
    )

    if args.simulate:
        # One-shot simulation using Dhan API data (fallback to sample data)
        print(f"\n📊 Running paper trading simulation ({args.days} days, {args.interval}min candles)...")
        df = fetch_nifty_intraday(interval=args.interval, days_back=args.days)
        if df.empty:
            print("  ⚠️  Dhan API unavailable — using sample data for simulation")
            timeframe_map = {1: "1min", 5: "5min", 15: "15min"}
            df = generate_sample_nifty_data(days=args.days, timeframe=timeframe_map.get(args.interval, "5min"))
        df = compute_indicators(df, fast_period=args.fast_ema, slow_period=args.slow_ema)
        engine.run_once(df)
    elif args.live:
        # Live paper trading with Dhan API
        print(f"\n🔴 LIVE PAPER TRADING MODE")
        print(f"   Data Source: Dhan API")
        print(f"   Interval: {args.interval}min")
        print(f"   Capital: ₹{args.capital:,.0f}")
        print(f"   Press Ctrl+C to stop\n")
        engine.run()
    else:
        # Live paper trading loop (default)
        engine.run()


if __name__ == "__main__":
    main()
