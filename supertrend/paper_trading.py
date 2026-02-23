"""
Supertrend + VWAP Scalping Paper Trading Engine
=================================================
Live paper trading using Supertrend, VWAP, and 9 EMA for high-accuracy scalping.
Polls Dhan API for real-time data, applies Supertrend + VWAP alignment,
manages trailing stops and targets.

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

# Ensure supertrend directory imports work
SUPERTREND_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(SUPERTREND_DIR)
sys.path.insert(0, SUPERTREND_DIR)
sys.path.insert(0, _ROOT_DIR)

# Dhan API lives in backtest/
BACKTEST_DIR = os.path.join(SUPERTREND_DIR, '..', 'backtest')

from config import (
    INITIAL_CAPITAL, NIFTY_LOT_SIZE, MAX_LOTS,
    TRADING_START, TRADING_END, SQUARE_OFF_TIME,
    SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER,
    EMA_PERIOD,
    MAX_SL_PER_TRADE, TARGET_MIN, TARGET_MAX, RISK_REWARD_RATIO,
    SL_BUFFER_POINTS, MIN_SL_POINTS, MAX_SL_POINTS,
    TRAIL_ACTIVATE_POINTS, TRAIL_LOCK_PCT,
    TIGHT_TRAIL_ACTIVATE, TIGHT_TRAIL_LOCK_PCT,
    ENTRY_COOLDOWN_BARS, MAX_TRADES_PER_DAY, MAX_LOSS_PER_DAY,
    INCLUDE_COSTS, BROKERAGE_PER_ORDER, SLIPPAGE_POINTS,
    STT_RATE, EXCHANGE_CHARGES, GST_RATE, SEBI_CHARGES, STAMP_DUTY,
    TIMEFRAME,
)
from data_utils import compute_indicators, generate_nifty_data
from strategy import Signal, TradeStatus, calculate_costs
from dhan_orders import DhanOrderManager

# Import Dhan API from backtest directory
import importlib.util
_dhan_spec = importlib.util.spec_from_file_location(
    "dhan_fetch", os.path.join(BACKTEST_DIR, "dhan_fetch.py")
)
_dhan_mod = importlib.util.module_from_spec(_dhan_spec)
_dhan_spec.loader.exec_module(_dhan_mod)
fetch_nifty_intraday = _dhan_mod.fetch_nifty_intraday
fetch_nifty_ltp = _dhan_mod.fetch_nifty_ltp
fetch_nifty_ohlc = _dhan_mod.fetch_nifty_ohlc

# ── Constants ────────────────────────────────────────────────
PAPER_DIR = os.path.join(SUPERTREND_DIR, "paper_trades")
STATE_FILE = os.path.join(PAPER_DIR, "supertrend_state.json")
TRADE_LOG_CSV = os.path.join(PAPER_DIR, "supertrend_trade_log.csv")
DAILY_LOG_CSV = os.path.join(PAPER_DIR, "supertrend_daily_summary.csv")

POLL_DELAY_SEC = 10
WARMUP_CANDLES = 20

# ── Logging ──────────────────────────────────────────────────
os.makedirs(PAPER_DIR, exist_ok=True)

logger = logging.getLogger("SupertrendPaper")
logger.setLevel(logging.DEBUG)

_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(_ch)

_fh = logging.FileHandler(os.path.join(PAPER_DIR, "supertrend_paper.log"), encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
logger.addHandler(_fh)


# ── Data Classes ─────────────────────────────────────────────
@dataclass
class PaperTrade:
    """A simulated scalping trade with full tracking."""
    trade_id: int
    entry_time: str
    direction: str                    # "LONG" or "SHORT"
    entry_price: float
    lots: int = MAX_LOTS
    sl_price: float = 0.0
    initial_sl_points: float = 0.0
    target_price: float = 0.0

    # Trailing stop
    trailing_sl: float = 0.0
    max_favorable: float = 0.0
    trail_active: bool = False
    tight_trail_active: bool = False

    # Final exit
    status: str = "OPEN"
    exit_time: str = ""
    exit_price: float = 0.0
    gross_pnl: float = 0.0
    costs: float = 0.0
    net_pnl: float = 0.0

    # Diagnostics
    entry_supertrend: float = 0.0
    entry_vwap: float = 0.0
    entry_ema: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.status == "OPEN"

    @property
    def quantity(self) -> int:
        return self.lots * NIFTY_LOT_SIZE

    def to_dict(self):
        d = asdict(self)
        d["quantity"] = self.quantity
        return d


# ── Supertrend Paper Trading Engine ──────────────────────────
class SupertrendPaperEngine:
    """
    Live paper trading engine for Supertrend + VWAP Scalping strategy.
    Polls Dhan API, applies Supertrend + VWAP + EMA signals, manages
    trailing stops and targets.
    """

    def __init__(self, interval: int = 5, capital: float = INITIAL_CAPITAL,
                 resume: bool = False, symbol: str = "nifty", real_mode: bool = False):
        self.interval = interval
        self.capital = capital
        self.initial_capital = capital
        self.symbol = symbol
        self.real_mode = real_mode
        self.order_mgr = DhanOrderManager(live=real_mode)

        self.current_trade: Optional[PaperTrade] = None
        self.closed_trades: List[PaperTrade] = []
        self.trade_counter = 0
        self.today = date.today()
        self.trades_today = 0
        self.daily_loss = 0.0
        self.daily_profit = 0.0
        self.last_exit_bar_idx = -ENTRY_COOLDOWN_BARS
        self.bar_counter = 0

        self.data: Optional[pd.DataFrame] = None
        self.last_bar_time: Optional[pd.Timestamp] = None
        self._running = False

        if resume:
            self._load_state()

    # ── Signal Generation ────────────────────────────────────
    def _generate_signal(self, row: pd.Series) -> str:
        """Generate entry signal from latest indicator row."""
        buy = row.get("buy_signal", False)
        sell = row.get("sell_signal", False)
        bullish = row.get("bullish_candle", False)
        bearish = row.get("bearish_candle", False)

        # Prefer strong candle confirmation
        if buy and bullish:
            return "LONG"
        if sell and bearish:
            return "SHORT"

        # Allow without strong candle if all 3 core conditions met
        if buy:
            return "LONG"
        if sell:
            return "SHORT"

        return "NONE"

    # ── Entry Logic ──────────────────────────────────────────
    def _should_enter(self, now: datetime) -> bool:
        """Check if conditions allow a new entry."""
        if self.current_trade and self.current_trade.is_open:
            return False
        if self.trades_today >= MAX_TRADES_PER_DAY:
            return False
        if self.daily_loss <= -MAX_LOSS_PER_DAY:
            return False
        if self.bar_counter - self.last_exit_bar_idx < ENTRY_COOLDOWN_BARS:
            return False

        time_str = f"{now.hour:02d}:{now.minute:02d}"
        if time_str < TRADING_START or time_str > TRADING_END:
            return False

        return True

    def _enter_trade(self, row: pd.Series, now: datetime, direction: str):
        """Enter a scalping trade."""
        price = row["close"]
        supertrend = row.get("supertrend", price)

        if pd.isna(supertrend) or supertrend <= 0:
            return

        # SL based on Supertrend
        if direction == "LONG":
            sl_price = supertrend - SL_BUFFER_POINTS
            sl_points = price - sl_price
        else:
            sl_price = supertrend + SL_BUFFER_POINTS
            sl_points = sl_price - price

        sl_points = max(MIN_SL_POINTS, min(sl_points, MAX_SL_POINTS))

        # Check max SL in rupees
        max_sl_points = MAX_SL_PER_TRADE / (MAX_LOTS * NIFTY_LOT_SIZE)
        if sl_points > max_sl_points:
            sl_points = max_sl_points

        if direction == "LONG":
            sl_price = price - sl_points
        else:
            sl_price = price + sl_points

        # Target
        target_points = sl_points * RISK_REWARD_RATIO
        target_inr = target_points * MAX_LOTS * NIFTY_LOT_SIZE
        if target_inr < TARGET_MIN:
            target_points = TARGET_MIN / (MAX_LOTS * NIFTY_LOT_SIZE)
        elif target_inr > TARGET_MAX:
            target_points = TARGET_MAX / (MAX_LOTS * NIFTY_LOT_SIZE)

        if direction == "LONG":
            target_price = price + target_points
        else:
            target_price = price - target_points

        self.trade_counter += 1
        now_str = str(now)

        self.current_trade = PaperTrade(
            trade_id=self.trade_counter,
            entry_time=now_str,
            direction=direction,
            entry_price=price,
            lots=MAX_LOTS,
            sl_price=sl_price,
            initial_sl_points=sl_points,
            target_price=target_price,
            trailing_sl=sl_price,
            entry_supertrend=supertrend,
            entry_vwap=row.get("vwap", 0),
            entry_ema=row.get("ema_9", 0),
        )
        self.trades_today += 1

        logger.info("=" * 60)
        logger.info(f"  🎯 SCALPING TRADE ENTERED  #{self.trade_counter}")
        logger.info(f"  Direction  : {direction}")
        logger.info(f"  Entry      : {price:.2f}")
        logger.info(f"  SL         : {sl_price:.2f} ({sl_points:.1f} pts = ₹{sl_points * MAX_LOTS * NIFTY_LOT_SIZE:,.0f})")
        logger.info(f"  Target     : {target_price:.2f} (₹{target_points * MAX_LOTS * NIFTY_LOT_SIZE:,.0f})")
        logger.info(f"  Supertrend : {supertrend:.2f}")
        logger.info(f"  VWAP       : {row.get('vwap', 0):.2f}")
        logger.info(f"  EMA-9      : {row.get('ema_9', 0):.2f}")
        logger.info("=" * 60)

        # Place real futures order
        txn = "BUY" if direction == "LONG" else "SELL"
        self.order_mgr.place_order(
            transaction_type=txn,
            symbol=self.symbol,
            quantity=self.current_trade.quantity,
            tag=f"supertrend_{self.trade_counter}_entry",
        )

        self._save_state()

    # ── Position Management ──────────────────────────────────
    def _manage_position(self, row: pd.Series, now: datetime):
        """Update position: check SL, target, trailing stop, supertrend flip, EOD."""
        trade = self.current_trade
        if not trade or not trade.is_open:
            return

        price = row["close"]
        high = row["high"]
        low = row["low"]
        now_str = str(now)
        time_str = f"{now.hour:02d}:{now.minute:02d}"

        # --- EOD Square-off ---
        if time_str >= SQUARE_OFF_TIME:
            self._close_trade(price, now_str, "TIME_EXIT")
            return

        if trade.direction == "LONG":
            # --- Stop Loss ---
            if low <= trade.sl_price:
                exit_price = trade.sl_price
                status = "SL_HIT" if not trade.trail_active else "TRAILING_SL"
                self._close_trade(exit_price, now_str, status)
                return

            # --- Target Hit ---
            if high >= trade.target_price:
                self._close_trade(trade.target_price, now_str, "TARGET_HIT")
                return

            # Track max favorable
            trade.max_favorable = max(trade.max_favorable, high - trade.entry_price)

            # --- Supertrend flip to red = exit ---
            st_dir = row.get("supertrend_direction", 1)
            if st_dir == -1:
                self._close_trade(price, now_str, "TRAILING_SL")
                return

            # --- Trailing Stop ---
            profit_pts = high - trade.entry_price
            if profit_pts >= TRAIL_ACTIVATE_POINTS and not trade.trail_active:
                trade.trail_active = True
            if profit_pts >= TIGHT_TRAIL_ACTIVATE and not trade.tight_trail_active:
                trade.tight_trail_active = True

            if trade.tight_trail_active:
                new_sl = high - profit_pts * (1 - TIGHT_TRAIL_LOCK_PCT)
                trade.sl_price = max(trade.sl_price, new_sl)
            elif trade.trail_active:
                new_sl = high - profit_pts * (1 - TRAIL_LOCK_PCT)
                trade.sl_price = max(trade.sl_price, new_sl)
            trade.trailing_sl = trade.sl_price

        else:  # SHORT
            if high >= trade.sl_price:
                exit_price = trade.sl_price
                status = "SL_HIT" if not trade.trail_active else "TRAILING_SL"
                self._close_trade(exit_price, now_str, status)
                return

            if low <= trade.target_price:
                self._close_trade(trade.target_price, now_str, "TARGET_HIT")
                return

            trade.max_favorable = max(trade.max_favorable, trade.entry_price - low)

            st_dir = row.get("supertrend_direction", -1)
            if st_dir == 1:
                self._close_trade(price, now_str, "TRAILING_SL")
                return

            profit_pts = trade.entry_price - low
            if profit_pts >= TRAIL_ACTIVATE_POINTS and not trade.trail_active:
                trade.trail_active = True
            if profit_pts >= TIGHT_TRAIL_ACTIVATE and not trade.tight_trail_active:
                trade.tight_trail_active = True

            if trade.tight_trail_active:
                new_sl = low + profit_pts * (1 - TIGHT_TRAIL_LOCK_PCT)
                trade.sl_price = min(trade.sl_price, new_sl)
            elif trade.trail_active:
                new_sl = low + profit_pts * (1 - TRAIL_LOCK_PCT)
                trade.sl_price = min(trade.sl_price, new_sl)
            trade.trailing_sl = trade.sl_price

        # Max daily loss check
        running_pnl = self._running_pnl(price)
        if self.daily_loss + min(0, running_pnl) <= -MAX_LOSS_PER_DAY:
            self._close_trade(price, now_str, "MAX_LOSS_EXIT")

    def _close_trade(self, exit_price: float, now_str: str, status: str):
        """Close position."""
        trade = self.current_trade
        if not trade:
            return

        trade.exit_price = exit_price
        trade.exit_time = now_str
        trade.status = status

        qty = trade.quantity

        if trade.direction == "LONG":
            trade.gross_pnl = (exit_price - trade.entry_price) * qty
        else:
            trade.gross_pnl = (trade.entry_price - exit_price) * qty

        entry_costs = calculate_costs(trade.entry_price, qty, is_entry=True)
        exit_costs = calculate_costs(exit_price, qty, is_entry=False)

        trade.costs = entry_costs["total"] + exit_costs["total"]
        trade.net_pnl = trade.gross_pnl - trade.costs

        self.capital += trade.net_pnl
        self.daily_loss += min(0, trade.net_pnl)
        self.daily_profit += max(0, trade.net_pnl)
        self.last_exit_bar_idx = self.bar_counter
        self.closed_trades.append(trade)
        self._log_trade(trade)

        emoji = {"TARGET_HIT": "🎯", "SL_HIT": "🛑", "TRAILING_SL": "📈",
                 "TIME_EXIT": "⏰", "MAX_LOSS_EXIT": "⚠️", "DAILY_TARGET_HIT": "💰"}
        icon = emoji.get(status, "❓")

        logger.info("=" * 60)
        logger.info(f"  {icon} TRADE CLOSED  #{trade.trade_id}  [{status}]")
        logger.info(f"  {trade.direction}: {trade.entry_price:.2f} -> {exit_price:.2f}")
        logger.info(f"  Gross P&L : {trade.gross_pnl:+,.0f}")
        logger.info(f"  Costs     : {trade.costs:,.0f}")
        logger.info(f"  Net P&L   : {trade.net_pnl:+,.0f}")
        logger.info(f"  Capital   : {self.capital:,.0f}")
        logger.info(f"  Daily P&L : {self.daily_profit + self.daily_loss:+,.0f}")
        logger.info("=" * 60)

        # Place exit order (opposite direction)
        exit_txn = "SELL" if trade.direction == "LONG" else "BUY"
        self.order_mgr.place_order(
            transaction_type=exit_txn,
            symbol=self.symbol,
            quantity=qty,
            tag=f"supertrend_{trade.trade_id}_close",
        )

        self.current_trade = None
        self._save_state()

    def _running_pnl(self, current_price: float) -> float:
        """Unrealized P&L of open trade."""
        if not self.current_trade or not self.current_trade.is_open:
            return 0
        trade = self.current_trade
        if trade.direction == "LONG":
            return (current_price - trade.entry_price) * trade.quantity
        else:
            return (trade.entry_price - current_price) * trade.quantity

    # ── Data Fetching ────────────────────────────────────────
    def _fetch_candles(self, days_back: int = 5) -> Optional[pd.DataFrame]:
        """Fetch Nifty intraday candles from Dhan API."""
        try:
            interval_map = {"1min": 1, "5min": 5, "15min": 15}
            interval = interval_map.get(TIMEFRAME, 5)
            df = fetch_nifty_intraday(interval=interval, days_back=days_back)

            if df is not None and not df.empty:
                logger.debug(f"Fetched {len(df)} candles from Dhan")
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

        while self._running:
            now = datetime.now()

            # Reset daily state
            if now.date() != self.today:
                self.today = now.date()
                self.trades_today = 0
                self.daily_loss = 0.0
                self.daily_profit = 0.0
                self.last_exit_bar_idx = -ENTRY_COOLDOWN_BARS
                self.bar_counter = 0
                logger.info(f"\n--- New Trading Day: {self.today} ---")

            # Market hours check
            market_open = now.replace(hour=9, minute=15, second=0)
            market_close = now.replace(hour=15, minute=30, second=0)

            if now < market_open:
                wait = (market_open - now).seconds
                logger.info(f"  Pre-market. Waiting {wait // 60}m...")
                time.sleep(min(wait, 300))
                continue

            if now > market_close:
                if self.current_trade and self.current_trade.is_open:
                    spot = fetch_nifty_ltp() or 0
                    if spot > 0:
                        self._close_trade(spot, str(now), "TIME_EXIT")

                self._print_daily_summary()
                logger.info("  Market closed. Waiting for next day...")
                time.sleep(3600)
                continue

            # Fetch and compute indicators
            df = self._fetch_candles(days_back=5)
            if df is None or df.empty:
                logger.warning("  No data, retrying in 30s...")
                time.sleep(30)
                continue

            df = compute_indicators(df)

            # Check for new bar
            latest_bar = df.index[-1]
            if self.last_bar_time and latest_bar <= self.last_bar_time:
                time.sleep(POLL_DELAY_SEC)
                continue
            self.last_bar_time = latest_bar
            self.bar_counter += 1

            row = df.iloc[-1]
            spot = row["close"]

            self._print_status(row, now)

            # Entry check
            if self._should_enter(now):
                signal = self._generate_signal(row)
                if signal != "NONE":
                    self._enter_trade(row, now, signal)

            # Manage open position
            if self.current_trade and self.current_trade.is_open:
                self._manage_position(row, now)

            time.sleep(self.interval * 60)

    # ── Simulate Mode ────────────────────────────────────────
    def run_simulate(self, days: int = 5):
        """Simulate paper trading on recent historical data."""
        self._print_banner("SIMULATE")
        logger.info(f"  Simulating {days} days of Supertrend scalping...")

        df = self._fetch_candles(days_back=days)
        if df is None or df.empty:
            logger.info("  Using sample data for simulation...")
            df = generate_nifty_data(days=days, timeframe=TIMEFRAME)

        df = compute_indicators(df)

        logger.info(f"  Data: {len(df)} bars | {df.index[0].date()} to {df.index[-1].date()}")
        logger.info(f"  Nifty range: {df['close'].min():.0f} - {df['close'].max():.0f}\n")

        df["_date"] = df.index.date
        for day_date, day_data in df.groupby("_date"):
            if len(day_data) < 15:
                continue

            self.today = day_date
            self.trades_today = 0
            self.daily_loss = 0.0
            self.daily_profit = 0.0
            self.last_exit_bar_idx = -ENTRY_COOLDOWN_BARS
            self.bar_counter = 0
            self.current_trade = None

            for ts in day_data.index:
                self.bar_counter += 1
                row = day_data.loc[ts]

                if self.current_trade and self.current_trade.is_open:
                    self._manage_position(row, ts)

                if self.current_trade is None or not self.current_trade.is_open:
                    if self._should_enter(ts):
                        signal = self._generate_signal(row)
                        if signal != "NONE":
                            self._enter_trade(row, ts, signal)

            # Force close at EOD
            if self.current_trade and self.current_trade.is_open:
                last_price = day_data["close"].iloc[-1]
                self._close_trade(last_price, str(day_data.index[-1]), "TIME_EXIT")

        self._print_final_summary()

    # ── Console Display ──────────────────────────────────────
    def _print_banner(self, mode: str):
        logger.info("\n" + "=" * 60)
        logger.info("  🎯 SUPERTREND + VWAP SCALPING PAPER TRADING")
        logger.info(f"  Mode: {mode} | Strategy: Supertrend + VWAP + 9 EMA")
        logger.info(f"  Capital: {self.capital:,.0f} | Max Lots: {MAX_LOTS}")
        logger.info(f"  Max SL: ₹{MAX_SL_PER_TRADE:,.0f}/trade | Target: ₹{TARGET_MIN:,.0f}–₹{TARGET_MAX:,.0f}")
        logger.info(f"  Max Trades/Day: {MAX_TRADES_PER_DAY} | Daily Target: ₹{5000:,.0f}")
        logger.info("=" * 60)

    def _print_status(self, row: pd.Series, now: datetime):
        trade = self.current_trade
        spot = row["close"]
        vwap = row.get("vwap", 0)
        st = row.get("supertrend", 0)
        st_dir = row.get("supertrend_direction", 0)
        ema = row.get("ema_9", 0)
        st_label = "🟢" if st_dir == 1 else "🔴"

        if trade and trade.is_open:
            if trade.direction == "LONG":
                upnl = (spot - trade.entry_price) * trade.quantity
            else:
                upnl = (trade.entry_price - spot) * trade.quantity
            icon = "+" if upnl >= 0 else ""
            logger.info(f"  {now.strftime('%H:%M')} | Nifty: {spot:.0f} | {trade.direction} @ {trade.entry_price:.0f} | SL: {trade.sl_price:.0f} | TGT: {trade.target_price:.0f} | P&L: {icon}{upnl:,.0f} | ST:{st_label}")
        else:
            logger.info(f"  {now.strftime('%H:%M')} | Nifty: {spot:.0f} | ST:{st_label} {st:.0f} | VWAP: {vwap:.0f} | EMA: {ema:.0f} | No position")

    def _print_daily_summary(self):
        today_trades = [t for t in self.closed_trades
                        if t.entry_time.startswith(str(self.today))]
        if not today_trades:
            logger.info("  No trades today.")
            return

        net = sum(t.net_pnl for t in today_trades)
        targets = sum(1 for t in today_trades if t.status == "TARGET_HIT")
        logger.info(f"\n  --- Daily Summary ({self.today}) ---")
        logger.info(f"  Trades: {len(today_trades)} | Net P&L: {'+'if net>0 else ''}{net:,.0f}")
        logger.info(f"  Targets hit: {targets}/{len(today_trades)}")
        logger.info(f"  Capital: {self.capital:,.0f}")

    def _print_final_summary(self):
        if not self.closed_trades:
            logger.info("  No trades executed.")
            return

        pnls = [t.net_pnl for t in self.closed_trades]
        wins = sum(1 for p in pnls if p > 0)
        total = len(pnls)
        targets = sum(1 for t in self.closed_trades if t.status == "TARGET_HIT")

        logger.info("\n" + "=" * 60)
        logger.info("  🎯 SUPERTREND SCALPING SIMULATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"  Total Trades:     {total}")
        logger.info(f"  Winners:          {wins} ({wins / total * 100:.1f}%)")
        logger.info(f"  Losers:           {total - wins} ({(total - wins) / total * 100:.1f}%)")
        logger.info(f"  Target Hits:      {targets}")
        logger.info(f"  Net P&L:          {sum(pnls):+,.0f}")
        logger.info(f"  Best Trade:       {max(pnls):+,.0f}")
        logger.info(f"  Worst Trade:      {min(pnls):+,.0f}")
        logger.info(f"  Capital:          {self.capital:,.0f}")
        logger.info("=" * 60)

    # ── Persistence ──────────────────────────────────────────
    def _save_state(self):
        state = {
            "capital": self.capital,
            "initial_capital": self.initial_capital,
            "trade_counter": self.trade_counter,
            "trades_today": self.trades_today,
            "daily_loss": self.daily_loss,
            "daily_profit": self.daily_profit,
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
            self.initial_capital = state.get("initial_capital", self.initial_capital)
            self.trade_counter = state.get("trade_counter", 0)
            self.trades_today = state.get("trades_today", 0)
            self.daily_loss = state.get("daily_loss", 0.0)
            self.daily_profit = state.get("daily_profit", 0.0)

            ct = state.get("current_trade")
            if ct:
                ct.pop("quantity", None)
                self.current_trade = PaperTrade(**ct)
                logger.info(f"  Resumed open trade #{self.current_trade.trade_id}")

            logger.info(f"  State restored. Capital: {self.capital:,.0f}")
        except Exception as e:
            logger.error(f"State load failed: {e}")

    def _log_trade(self, trade: PaperTrade):
        """Append trade to CSV log."""
        row = {
            "trade_id": trade.trade_id,
            "date": trade.entry_time[:10],
            "direction": trade.direction,
            "entry_price": round(trade.entry_price, 2),
            "exit_price": round(trade.exit_price, 2),
            "lots": trade.lots,
            "sl_price": round(trade.sl_price, 2),
            "target_price": round(trade.target_price, 2),
            "gross_pnl": round(trade.gross_pnl, 2),
            "costs": round(trade.costs, 2),
            "net_pnl": round(trade.net_pnl, 2),
            "status": trade.status,
            "entry_supertrend": round(trade.entry_supertrend, 2),
            "entry_vwap": round(trade.entry_vwap, 2),
            "entry_ema": round(trade.entry_ema, 2),
            "max_favorable": round(trade.max_favorable, 2),
            "capital": round(self.capital, 2),
        }
        df = pd.DataFrame([row])
        header = not os.path.exists(TRADE_LOG_CSV)
        df.to_csv(TRADE_LOG_CSV, mode="a", header=header, index=False)


# ── CLI ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Supertrend + VWAP Scalping Paper Trading Engine"
    )
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

    engine = SupertrendPaperEngine(
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
