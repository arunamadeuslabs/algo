"""
Sapphire Paper Trading Engine
===============================
Live paper trading for the Nifty Sapphire short strangle strategy.
Polls Dhan API for real-time data, sells OTM CE+PE strangle at 09:20,
manages trailing SL on both legs, and exits at 15:25 or on SL hit.

Usage:
  python paper_trading.py --live                  # Live paper trading
  python paper_trading.py --simulate --days 5     # Simulate on recent data
  python paper_trading.py --live --resume          # Resume from saved state
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

# Ensure sapphire directory imports work FIRST
SAPPHIRE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(SAPPHIRE_DIR)
sys.path.insert(0, SAPPHIRE_DIR)
sys.path.insert(0, _ROOT_DIR)

from config import (
    INITIAL_CAPITAL, OPTION_LOT_SIZE, NUM_LOTS,
    ENTRY_TIME, SQUARE_OFF_TIME, TRADING_START, TRADING_END,
    OTM_OFFSET_CE, OTM_OFFSET_PE, STRIKE_ROUNDING, DAYS_TO_EXPIRY,
    INITIAL_SL_PCT, TRAIL_ACTIVATE_PCT, TRAIL_LOCK_PCT,
    DEEP_PROFIT_PCT, DEEP_TRAIL_LOCK_PCT,
    COMBINED_SL_PCT, MAX_LOSS_PER_DAY,
    MOMENTUM_THRESHOLD, MOMENTUM_SHIFT_ENABLED,
    INCLUDE_COSTS, BROKERAGE_PER_ORDER, SLIPPAGE_POINTS,
    STT_RATE, EXCHANGE_CHARGES, GST_RATE, SEBI_CHARGES, STAMP_DUTY,
    NIFTY_IV, RISK_FREE_RATE, TIMEFRAME,
)
from data_utils import (
    black_scholes_price, get_atm_strike, get_otm_strikes, generate_nifty_data,
)
from strategy import calculate_costs, LegStatus
import dhan_api
from dhan_orders import DhanOrderManager

# ── Constants ────────────────────────────────────────────────
PAPER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_trades")
STATE_FILE = os.path.join(PAPER_DIR, "sapphire_state.json")
TRADE_LOG_CSV = os.path.join(PAPER_DIR, "sapphire_trade_log.csv")
DAILY_LOG_CSV = os.path.join(PAPER_DIR, "sapphire_daily_summary.csv")

POLL_DELAY_SEC = 10
WARMUP_CANDLES = 30

# ── Logging ──────────────────────────────────────────────────
os.makedirs(PAPER_DIR, exist_ok=True)

logger = logging.getLogger("SapphirePaper")
logger.setLevel(logging.DEBUG)

_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(_ch)

_fh = logging.FileHandler(os.path.join(PAPER_DIR, "sapphire_paper.log"), encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
logger.addHandler(_fh)


# ── Data Classes ─────────────────────────────────────────────
@dataclass
class PaperLeg:
    """One leg of the paper strangle."""
    option_type: str          # "CE" or "PE"
    strike: float
    entry_premium: float
    current_premium: float = 0.0
    exit_premium: float = 0.0
    sl_premium: float = 0.0
    min_premium_seen: float = 999.0
    trail_active: bool = False
    deep_trail_active: bool = False
    status: str = "OPEN"      # OPEN, SL_HIT, EOD, COMBINED_SL, MAX_LOSS
    entry_time: str = ""
    exit_time: str = ""
    gross_pnl: float = 0.0
    costs: float = 0.0
    net_pnl: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.status == "OPEN"

    def to_dict(self):
        return asdict(self)


@dataclass
class PaperStrangle:
    """One day's paper strangle."""
    trade_id: int
    date: str
    entry_spot: float
    ce_leg: Optional[PaperLeg] = None
    pe_leg: Optional[PaperLeg] = None
    entry_time: str = ""
    exit_time: str = ""
    entry_combined_prem: float = 0.0
    gross_pnl: float = 0.0
    total_costs: float = 0.0
    net_pnl: float = 0.0
    exit_reason: str = ""
    momentum_detected: bool = False
    spot_at_exit: float = 0.0

    @property
    def is_open(self) -> bool:
        if self.ce_leg and self.ce_leg.is_open:
            return True
        if self.pe_leg and self.pe_leg.is_open:
            return True
        return False

    def to_dict(self):
        d = asdict(self)
        return d


# ── Sapphire Paper Trading Engine ────────────────────────────
class SapphirePaperEngine:
    """
    Live paper trading engine for Sapphire short strangle.
    Polls Dhan API every 5min, manages strangle positions with trailing SL.
    """

    def __init__(self, interval: int = 5, capital: float = INITIAL_CAPITAL,
                 resume: bool = False, symbol: str = "nifty", real_mode: bool = False):
        self.interval = interval
        self.capital = capital
        self.initial_capital = capital
        self.quantity = OPTION_LOT_SIZE * NUM_LOTS
        self.symbol = symbol
        self.real_mode = real_mode
        self.order_mgr = DhanOrderManager(live=real_mode)

        self.current_trade: Optional[PaperStrangle] = None
        self.closed_trades: List[PaperStrangle] = []
        self.trade_counter = 0
        self.today = date.today()
        self.entered_today = False

        self.data: Optional[pd.DataFrame] = None
        self.last_bar_time: Optional[pd.Timestamp] = None
        self._running = False

        if resume:
            self._load_state()

    # ── Entry Logic ──────────────────────────────────────────
    def _should_enter(self, now: datetime) -> bool:
        """Check if we should enter a strangle now."""
        if self.current_trade and self.current_trade.is_open:
            return False    # Already in a position
        if self.entered_today:
            return False    # One trade per day

        time_str = f"{now.hour:02d}:{now.minute:02d}"
        if time_str < ENTRY_TIME or time_str > "09:35":
            return False    # Only enter 09:20-09:35 window

        return True

    def _enter_strangle(self, spot: float, now: datetime):
        """Enter strangle: sell OTM CE + OTM PE."""
        ce_strike, pe_strike = get_otm_strikes(
            spot, OTM_OFFSET_CE, OTM_OFFSET_PE, STRIKE_ROUNDING
        )

        iv = NIFTY_IV / 100.0
        r = RISK_FREE_RATE / 100.0
        dte = DAYS_TO_EXPIRY / 365.0

        ce_prem = black_scholes_price(spot, ce_strike, dte, iv, r, "CE")
        pe_prem = black_scholes_price(spot, pe_strike, dte, iv, r, "PE")

        ce_prem = max(ce_prem, 2.0)
        pe_prem = max(pe_prem, 2.0)

        self.trade_counter += 1
        now_str = str(now)

        ce_leg = PaperLeg(
            option_type="CE",
            strike=ce_strike,
            entry_premium=ce_prem,
            current_premium=ce_prem,
            sl_premium=ce_prem * (1 + INITIAL_SL_PCT),
            min_premium_seen=ce_prem,
            entry_time=now_str,
        )

        pe_leg = PaperLeg(
            option_type="PE",
            strike=pe_strike,
            entry_premium=pe_prem,
            current_premium=pe_prem,
            sl_premium=pe_prem * (1 + INITIAL_SL_PCT),
            min_premium_seen=pe_prem,
            entry_time=now_str,
        )

        self.current_trade = PaperStrangle(
            trade_id=self.trade_counter,
            date=str(now.date()) if hasattr(now, 'date') else str(self.today),
            entry_spot=spot,
            ce_leg=ce_leg,
            pe_leg=pe_leg,
            entry_time=now_str,
            entry_combined_prem=ce_prem + pe_prem,
        )
        self.entered_today = True

        logger.info("=" * 60)
        logger.info(f"  STRANGLE ENTERED  #{self.trade_counter}")
        logger.info(f"  Spot: {spot:.2f}")
        logger.info(f"  CE: {ce_strike} @ {ce_prem:.2f}  (SL: {ce_leg.sl_premium:.2f})")
        logger.info(f"  PE: {pe_strike} @ {pe_prem:.2f}  (SL: {pe_leg.sl_premium:.2f})")
        logger.info(f"  Combined Premium: {ce_prem + pe_prem:.2f}")
        logger.info("=" * 60)

        # Place real orders if in live mode
        self.order_mgr.place_order(
            transaction_type="SELL",
            symbol=self.symbol,
            quantity=self.quantity,
            strike=ce_strike,
            option_type="CE",
            tag=f"sapphire_{self.trade_counter}_ce_entry",
        )
        self.order_mgr.place_order(
            transaction_type="SELL",
            symbol=self.symbol,
            quantity=self.quantity,
            strike=pe_strike,
            option_type="PE",
            tag=f"sapphire_{self.trade_counter}_pe_entry",
        )

        self._save_state()

    # ── Position Management ──────────────────────────────────
    def _manage_position(self, spot: float, high: float, low: float, now: datetime):
        """Update premiums, check trailing SL, manage both legs."""
        trade = self.current_trade
        if not trade or not trade.is_open:
            return

        iv = NIFTY_IV / 100.0
        r = RISK_FREE_RATE / 100.0

        # Time decay calculation
        minutes_from_open = (now.hour - 9) * 60 + (now.minute - 15)
        day_fraction = max(0, min(minutes_from_open / 375, 1.0))
        remaining_dte = max(DAYS_TO_EXPIRY - day_fraction, 0.01) / 365.0

        # IV adjustment based on spot movement
        spot_move_pct = abs(spot - trade.entry_spot) / trade.entry_spot * 100
        iv_adj = iv
        if spot_move_pct > 1.0:
            iv_adj *= (1 + spot_move_pct * 0.25)
        elif spot_move_pct > 0.5:
            iv_adj *= (1 + spot_move_pct * 0.15)

        now_str = str(now)
        ce = trade.ce_leg
        pe = trade.pe_leg

        # Update CE leg
        if ce.is_open:
            ce.current_premium = black_scholes_price(spot, ce.strike, remaining_dte, iv_adj, r, "CE")
            ce_worst = black_scholes_price(high, ce.strike, remaining_dte, iv_adj, r, "CE")
            ce.min_premium_seen = min(ce.min_premium_seen, ce.current_premium)

            self._update_trailing_sl(ce)
            if ce_worst >= ce.sl_premium:
                ce.exit_premium = ce.sl_premium
                ce.exit_time = now_str
                ce.status = "SL_HIT"
                logger.info(f"  CE SL HIT @ {ce.sl_premium:.2f} (entry: {ce.entry_premium:.2f})")
                # Buy back CE on SL
                self.order_mgr.place_order(
                    transaction_type="BUY",
                    symbol=self.symbol,
                    quantity=self.quantity,
                    strike=ce.strike,
                    option_type="CE",
                    tag=f"sapphire_{trade.trade_id}_ce_sl",
                )

        # Update PE leg
        if pe.is_open:
            pe.current_premium = black_scholes_price(spot, pe.strike, remaining_dte, iv_adj, r, "PE")
            pe_worst = black_scholes_price(low, pe.strike, remaining_dte, iv_adj, r, "PE")
            pe.min_premium_seen = min(pe.min_premium_seen, pe.current_premium)

            self._update_trailing_sl(pe)
            if pe_worst >= pe.sl_premium:
                pe.exit_premium = pe.sl_premium
                pe.exit_time = now_str
                pe.status = "SL_HIT"
                logger.info(f"  PE SL HIT @ {pe.sl_premium:.2f} (entry: {pe.entry_premium:.2f})")
                # Buy back PE on SL
                self.order_mgr.place_order(
                    transaction_type="BUY",
                    symbol=self.symbol,
                    quantity=self.quantity,
                    strike=pe.strike,
                    option_type="PE",
                    tag=f"sapphire_{trade.trade_id}_pe_sl",
                )

        # Combined SL check
        if ce.is_open or pe.is_open:
            current_combined = (
                (ce.current_premium if ce.is_open else ce.exit_premium) +
                (pe.current_premium if pe.is_open else pe.exit_premium)
            )
            if current_combined > trade.entry_combined_prem * (1 + COMBINED_SL_PCT):
                if ce.is_open:
                    ce.exit_premium = ce.current_premium
                    ce.exit_time = now_str
                    ce.status = "COMBINED_SL"
                    self.order_mgr.place_order(
                        transaction_type="BUY", symbol=self.symbol,
                        quantity=self.quantity, strike=ce.strike, option_type="CE",
                        tag=f"sapphire_{trade.trade_id}_ce_combsl",
                    )
                if pe.is_open:
                    pe.exit_premium = pe.current_premium
                    pe.exit_time = now_str
                    pe.status = "COMBINED_SL"
                    self.order_mgr.place_order(
                        transaction_type="BUY", symbol=self.symbol,
                        quantity=self.quantity, strike=pe.strike, option_type="PE",
                        tag=f"sapphire_{trade.trade_id}_pe_combsl",
                    )
                trade.exit_reason = "Combined SL"
                logger.info(f"  COMBINED SL triggered! Combined prem: {current_combined:.2f} > {trade.entry_combined_prem * (1 + COMBINED_SL_PCT):.2f}")

        # Max daily loss check
        running_pnl = self._running_pnl(trade)
        if running_pnl < -MAX_LOSS_PER_DAY:
            if ce.is_open:
                ce.exit_premium = ce.current_premium
                ce.exit_time = now_str
                ce.status = "MAX_LOSS"
                self.order_mgr.place_order(
                    transaction_type="BUY", symbol=self.symbol,
                    quantity=self.quantity, strike=ce.strike, option_type="CE",
                    tag=f"sapphire_{trade.trade_id}_ce_maxloss",
                )
            if pe.is_open:
                pe.exit_premium = pe.current_premium
                pe.exit_time = now_str
                pe.status = "MAX_LOSS"
                self.order_mgr.place_order(
                    transaction_type="BUY", symbol=self.symbol,
                    quantity=self.quantity, strike=pe.strike, option_type="PE",
                    tag=f"sapphire_{trade.trade_id}_pe_maxloss",
                )
            trade.exit_reason = "Max Daily Loss"
            logger.info(f"  MAX DAILY LOSS triggered! P&L: {running_pnl:.0f}")

        # Momentum detection
        if MOMENTUM_SHIFT_ENABLED and abs(spot - trade.entry_spot) >= MOMENTUM_THRESHOLD:
            trade.momentum_detected = True

        # EOD square-off
        time_str = f"{now.hour:02d}:{now.minute:02d}"
        if time_str >= SQUARE_OFF_TIME:
            self._square_off(now_str, spot)

        # If both legs closed, finalize trade
        if not trade.is_open:
            self._finalize_trade(spot, now_str)

    def _update_trailing_sl(self, leg: PaperLeg):
        """Dynamic trailing SL: Phase 1 → Phase 2 → Phase 3."""
        entry = leg.entry_premium
        decay_pct = (entry - leg.min_premium_seen) / entry if entry > 0 else 0

        # Phase 3: Deep profit
        if decay_pct >= DEEP_PROFIT_PCT and not leg.deep_trail_active:
            leg.deep_trail_active = True
            leg.trail_active = True

        if leg.deep_trail_active:
            max_gain = entry - leg.min_premium_seen
            new_sl = entry - (max_gain * DEEP_TRAIL_LOCK_PCT)
            leg.sl_premium = min(leg.sl_premium, max(new_sl, leg.min_premium_seen * 1.15))
            return

        # Phase 2: Standard trailing
        if decay_pct >= TRAIL_ACTIVATE_PCT and not leg.trail_active:
            leg.trail_active = True

        if leg.trail_active:
            max_gain = entry - leg.min_premium_seen
            new_sl = entry - (max_gain * TRAIL_LOCK_PCT)
            leg.sl_premium = min(leg.sl_premium, max(new_sl, leg.min_premium_seen * 1.25))

    def _square_off(self, now_str: str, spot: float):
        """Square off all open legs at EOD."""
        trade = self.current_trade
        if not trade:
            return
        if trade.ce_leg.is_open:
            trade.ce_leg.exit_premium = trade.ce_leg.current_premium
            trade.ce_leg.exit_time = now_str
            trade.ce_leg.status = "EOD"
            # Buy back CE leg
            self.order_mgr.place_order(
                transaction_type="BUY",
                symbol=self.symbol,
                quantity=self.quantity,
                strike=trade.ce_leg.strike,
                option_type="CE",
                tag=f"sapphire_{trade.trade_id}_ce_squareoff",
            )
        if trade.pe_leg.is_open:
            trade.pe_leg.exit_premium = trade.pe_leg.current_premium
            trade.pe_leg.exit_time = now_str
            trade.pe_leg.status = "EOD"
            # Buy back PE leg
            self.order_mgr.place_order(
                transaction_type="BUY",
                symbol=self.symbol,
                quantity=self.quantity,
                strike=trade.pe_leg.strike,
                option_type="PE",
                tag=f"sapphire_{trade.trade_id}_pe_squareoff",
            )
        if not trade.exit_reason:
            trade.exit_reason = "EOD Square-off"

    def _finalize_trade(self, spot: float, now_str: str):
        """Compute P&L and close the trade."""
        trade = self.current_trade
        ce = trade.ce_leg
        pe = trade.pe_leg

        ce.gross_pnl = (ce.entry_premium - ce.exit_premium) * self.quantity
        pe.gross_pnl = (pe.entry_premium - pe.exit_premium) * self.quantity

        ce_entry_cost = calculate_costs(ce.entry_premium, self.quantity, True)
        ce_exit_cost = calculate_costs(ce.exit_premium, self.quantity, False)
        pe_entry_cost = calculate_costs(pe.entry_premium, self.quantity, True)
        pe_exit_cost = calculate_costs(pe.exit_premium, self.quantity, False)

        ce.costs = ce_entry_cost["total"] + ce_exit_cost["total"]
        pe.costs = pe_entry_cost["total"] + pe_exit_cost["total"]
        ce.net_pnl = ce.gross_pnl - ce.costs
        pe.net_pnl = pe.gross_pnl - pe.costs

        trade.gross_pnl = ce.gross_pnl + pe.gross_pnl
        trade.total_costs = ce.costs + pe.costs
        trade.net_pnl = trade.gross_pnl - trade.total_costs
        trade.spot_at_exit = spot
        trade.exit_time = now_str

        if not trade.exit_reason:
            if ce.status == "SL_HIT" or pe.status == "SL_HIT":
                trade.exit_reason = "Trail SL Hit"
            else:
                trade.exit_reason = "EOD Square-off"

        self.capital += trade.net_pnl
        self.closed_trades.append(trade)
        self._log_trade(trade)

        result_icon = "+" if trade.net_pnl > 0 else ""
        logger.info("=" * 60)
        logger.info(f"  STRANGLE CLOSED  #{trade.trade_id}")
        logger.info(f"  CE: {ce.strike} | {ce.entry_premium:.2f} -> {ce.exit_premium:.2f} | {ce.status}")
        logger.info(f"  PE: {pe.strike} | {pe.entry_premium:.2f} -> {pe.exit_premium:.2f} | {pe.status}")
        logger.info(f"  Gross P&L: {result_icon}{trade.gross_pnl:,.0f}")
        logger.info(f"  Costs:     {trade.total_costs:,.0f}")
        logger.info(f"  Net P&L:   {result_icon}{trade.net_pnl:,.0f}  ({trade.exit_reason})")
        logger.info(f"  Capital:   {self.capital:,.0f}")
        logger.info("=" * 60)

        self.current_trade = None
        self._save_state()

    def _running_pnl(self, trade: PaperStrangle) -> float:
        """Unrealized + realized P&L."""
        pnl = 0
        ce, pe = trade.ce_leg, trade.pe_leg
        if ce.is_open:
            pnl += (ce.entry_premium - ce.current_premium) * self.quantity
        else:
            pnl += ce.gross_pnl
        if pe.is_open:
            pnl += (pe.entry_premium - pe.current_premium) * self.quantity
        else:
            pnl += pe.gross_pnl
        return pnl

    # ── Data Fetching ────────────────────────────────────────
    def _fetch_candles(self, days_back: int = 5) -> Optional[pd.DataFrame]:
        """Fetch intraday candles from Dhan API."""
        try:
            interval_map = {"1min": 1, "5min": 5, "15min": 15}
            interval = interval_map.get(TIMEFRAME, 5)
            df = dhan_api.fetch_intraday(self.symbol, interval=interval, days_back=days_back)

            if df is not None and not df.empty:
                logger.debug(f"Fetched {len(df)} candles from Dhan ({df.index[0]} to {df.index[-1]})")
                return df
            else:
                logger.warning("Dhan returned empty data")
                return None
        except Exception as e:
            logger.error(f"Dhan fetch error: {e}")
            return None

    # ── Live Mode ────────────────────────────────────────────
    def run_live(self):
        """Run live paper trading during market hours."""
        self._running = True

        def _stop(signum, frame):
            self._running = False
            logger.info("Shutdown signal received...")

        sig.signal(sig.SIGINT, _stop)

        self._print_banner("LIVE")
        logger.info(f"  Polling Dhan every {self.interval}min | Capital: {self.capital:,.0f}")
        logger.info(f"  Strangle: CE+{OTM_OFFSET_CE} / PE-{OTM_OFFSET_PE} OTM")
        logger.info(f"  Trailing SL: {INITIAL_SL_PCT*100:.0f}% init -> {TRAIL_LOCK_PCT*100:.0f}% trail -> {DEEP_TRAIL_LOCK_PCT*100:.0f}% deep")

        while self._running:
            now = datetime.now()

            # Reset daily state
            if now.date() != self.today:
                self.today = now.date()
                self.entered_today = False
                logger.info(f"\n--- New Trading Day: {self.today} ---")

            # Market hours check (9:15 - 15:30)
            market_open = now.replace(hour=9, minute=15, second=0)
            market_close = now.replace(hour=15, minute=30, second=0)

            if now < market_open:
                wait = (market_open - now).seconds
                logger.info(f"  Pre-market. Waiting {wait//60}m for market open...")
                time.sleep(min(wait, 300))
                continue

            if now > market_close:
                # EOD: square off if still open
                if self.current_trade and self.current_trade.is_open:
                    spot = dhan_api.fetch_ltp(self.symbol) or 0
                    if spot > 0:
                        self._square_off(str(now), spot)
                        self._finalize_trade(spot, str(now))

                self._print_daily_summary()
                logger.info("  Market closed. Waiting for next day...")
                time.sleep(3600)
                continue

            # Fetch latest data
            df = self._fetch_candles(days_back=3)
            if df is None or df.empty:
                logger.warning("  No data, retrying in 30s...")
                time.sleep(30)
                continue

            # Check for new bar
            latest_bar = df.index[-1]
            if self.last_bar_time and latest_bar <= self.last_bar_time:
                time.sleep(POLL_DELAY_SEC)
                continue
            self.last_bar_time = latest_bar

            spot = df["close"].iloc[-1]
            high = df["high"].iloc[-1]
            low = df["low"].iloc[-1]

            # Print status
            self._print_status(spot, now)

            # Entry check
            if self._should_enter(now):
                self._enter_strangle(spot, now)

            # Manage open position
            if self.current_trade and self.current_trade.is_open:
                self._manage_position(spot, high, low, now)

            # Wait for next candle
            time.sleep(self.interval * 60)

    # ── Simulate Mode ────────────────────────────────────────
    def run_simulate(self, days: int = 5):
        """Simulate paper trading on recent historical data."""
        self._print_banner("SIMULATE")
        logger.info(f"  Simulating {days} days of strangle trading...")

        # Try Dhan API first, fallback to sample data
        df = self._fetch_candles(days_back=days)
        if df is None or df.empty:
            logger.info("  Using sample data for simulation...")
            df = generate_nifty_data(days=days, timeframe=TIMEFRAME)

        logger.info(f"  Data: {len(df)} bars | {df.index[0].date()} to {df.index[-1].date()}")
        logger.info(f"  Nifty range: {df['close'].min():.0f} - {df['close'].max():.0f}\n")

        # Group by day
        df["_date"] = df.index.date
        for day_date, day_data in df.groupby("_date"):
            if len(day_data) < 10:
                continue

            self.today = day_date
            self.entered_today = False
            self.current_trade = None

            for ts in day_data.index:
                spot = day_data.loc[ts, "close"]
                high = day_data.loc[ts, "high"]
                low = day_data.loc[ts, "low"]

                if self._should_enter(ts):
                    self._enter_strangle(spot, ts)

                if self.current_trade and self.current_trade.is_open:
                    self._manage_position(spot, high, low, ts)

            # Force close if still open
            if self.current_trade and self.current_trade.is_open:
                last_spot = day_data["close"].iloc[-1]
                self._square_off(str(day_data.index[-1]), last_spot)
                self._finalize_trade(last_spot, str(day_data.index[-1]))

        self._print_final_summary()

    # ── Console Display ──────────────────────────────────────
    def _print_banner(self, mode: str):
        logger.info("\n" + "=" * 60)
        logger.info("  SAPPHIRE PAPER TRADING")
        logger.info(f"  Mode: {mode} | Strategy: Short Strangle")
        logger.info("=" * 60)

    def _print_status(self, spot: float, now: datetime):
        trade = self.current_trade
        if trade and trade.is_open:
            rpnl = self._running_pnl(trade)
            icon = "+" if rpnl >= 0 else ""
            ce_info = f"CE {trade.ce_leg.strike}: {trade.ce_leg.current_premium:.1f}" if trade.ce_leg.is_open else f"CE CLOSED"
            pe_info = f"PE {trade.pe_leg.strike}: {trade.pe_leg.current_premium:.1f}" if trade.pe_leg.is_open else f"PE CLOSED"
            logger.info(f"  {now.strftime('%H:%M')} | Nifty: {spot:.0f} | {ce_info} | {pe_info} | P&L: {icon}{rpnl:,.0f}")
        else:
            logger.info(f"  {now.strftime('%H:%M')} | Nifty: {spot:.0f} | No position")

    def _print_daily_summary(self):
        today_trades = [t for t in self.closed_trades if t.date == str(self.today)]
        if not today_trades:
            logger.info("  No trades today.")
            return

        net = sum(t.net_pnl for t in today_trades)
        logger.info(f"\n  --- Daily Summary ({self.today}) ---")
        logger.info(f"  Trades: {len(today_trades)} | Net P&L: {'+'if net>0 else ''}{net:,.0f}")
        logger.info(f"  Capital: {self.capital:,.0f}")
        for t in today_trades:
            logger.info(f"    #{t.trade_id}: CE {t.ce_leg.strike}/{t.pe_leg.strike} PE | {t.exit_reason} | {'+'if t.net_pnl>0 else ''}{t.net_pnl:,.0f}")

    def _print_final_summary(self):
        if not self.closed_trades:
            logger.info("  No trades executed.")
            return

        pnls = [t.net_pnl for t in self.closed_trades]
        wins = sum(1 for p in pnls if p > 0)
        total = len(pnls)

        logger.info("\n" + "=" * 60)
        logger.info("  SAPPHIRE SIMULATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"  Total Trades:   {total}")
        logger.info(f"  Winners:        {wins} ({wins/total*100:.1f}%)")
        logger.info(f"  Losers:         {total - wins} ({(total-wins)/total*100:.1f}%)")
        logger.info(f"  Gross P&L:      {sum(t.gross_pnl for t in self.closed_trades):+,.0f}")
        logger.info(f"  Total Costs:    {sum(t.total_costs for t in self.closed_trades):,.0f}")
        logger.info(f"  Net P&L:        {sum(pnls):+,.0f}")
        logger.info(f"  Best Trade:     {max(pnls):+,.0f}")
        logger.info(f"  Worst Trade:    {min(pnls):+,.0f}")
        logger.info(f"  Capital:        {self.capital:,.0f}")
        logger.info("=" * 60)

    # ── Persistence ──────────────────────────────────────────
    def _save_state(self):
        state = {
            "capital": self.capital,
            "trade_counter": self.trade_counter,
            "entered_today": self.entered_today,
            "today": str(self.today),
            "current_trade": self.current_trade.to_dict() if self.current_trade else None,
        }
        try:
            with open(STATE_FILE, "w") as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"State save failed: {e}")

    def _load_state(self):
        if not os.path.exists(STATE_FILE):
            logger.info("  No saved state found.")
            return
        try:
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
            self.capital = state.get("capital", self.capital)
            self.trade_counter = state.get("trade_counter", 0)
            self.entered_today = state.get("entered_today", False)
            logger.info(f"  Resumed state: Capital={self.capital:,.0f}, Trades={self.trade_counter}")
        except Exception as e:
            logger.error(f"State load failed: {e}")

    def _log_trade(self, trade: PaperStrangle):
        """Append trade to CSV log."""
        row = {
            "trade_id": trade.trade_id,
            "date": trade.date,
            "entry_spot": round(trade.entry_spot, 2),
            "exit_spot": round(trade.spot_at_exit, 2),
            "ce_strike": trade.ce_leg.strike,
            "pe_strike": trade.pe_leg.strike,
            "ce_entry_prem": round(trade.ce_leg.entry_premium, 2),
            "pe_entry_prem": round(trade.pe_leg.entry_premium, 2),
            "ce_exit_prem": round(trade.ce_leg.exit_premium, 2),
            "pe_exit_prem": round(trade.pe_leg.exit_premium, 2),
            "gross_pnl": round(trade.gross_pnl, 2),
            "costs": round(trade.total_costs, 2),
            "net_pnl": round(trade.net_pnl, 2),
            "exit_reason": trade.exit_reason,
            "ce_status": trade.ce_leg.status,
            "pe_status": trade.pe_leg.status,
            "capital": round(self.capital, 2),
        }
        df = pd.DataFrame([row])
        header = not os.path.exists(TRADE_LOG_CSV)
        df.to_csv(TRADE_LOG_CSV, mode="a", header=header, index=False)


# ── CLI ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Sapphire Paper Trading Engine")
    parser.add_argument("--live", action="store_true", help="Run live paper trading")
    parser.add_argument("--simulate", action="store_true", help="Simulate on recent data")
    parser.add_argument("--days", type=int, default=5, help="Days to simulate (default: 5)")
    parser.add_argument("--interval", type=int, default=5, help="Candle interval in minutes")
    parser.add_argument("--capital", type=float, default=INITIAL_CAPITAL, help="Starting capital")
    parser.add_argument("--resume", action="store_true", help="Resume from saved state")
    parser.add_argument("--symbol", type=str, default="nifty",
                        choices=["nifty", "banknifty", "finnifty", "midcapnifty", "sensex"],
                        help="Index to trade (default: nifty)")
    parser.add_argument("--real", action="store_true",
                        help="LIVE TRADING: Place real orders via Dhan API")

    args = parser.parse_args()

    engine = SapphirePaperEngine(
        interval=args.interval,
        capital=args.capital,
        resume=args.resume,
        symbol=args.symbol,
        real_mode=args.real,
    )

    if args.live or args.real:
        engine.run_live()
    elif args.simulate:
        engine.run_simulate(days=args.days)
    else:
        print("Usage: python paper_trading.py --live|--simulate [--days N]")
        print("  --live       Live paper trading (polls Dhan API)")
        print("  --simulate   Simulate on historical/sample data")
        print("  --days N     Simulation period (default: 5)")
        print("  --resume     Resume from saved state")


if __name__ == "__main__":
    main()
