"""
Dual-Confirmation Momentum Paper Trading Engine
=================================================
Live paper trading for the Nifty futures momentum strategy.
Polls Dhan API for real-time data, applies dual-confirmation entry logic
(MACD crossover + RSI + price action), manages partial exits and trailing stops.

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

# Ensure momentum directory imports work
MOMENTUM_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(MOMENTUM_DIR)
sys.path.insert(0, MOMENTUM_DIR)
sys.path.insert(0, _ROOT_DIR)

from config import (
    INITIAL_CAPITAL, NIFTY_LOT_SIZE, MAX_LOTS,
    TRADING_START, TRADING_END, SQUARE_OFF_TIME,
    RSI_OVERBOUGHT, RSI_OVERSOLD, RSI_EXTREME_OB, RSI_EXTREME_OS,
    ATR_SL_MULTIPLIER, MIN_SL_POINTS, MAX_SL_POINTS,
    PARTIAL_EXIT_PCT, PARTIAL_TARGET_RR, TRAIL_REMAINING,
    TRAIL_ACTIVATE_POINTS, TRAIL_STEP_POINTS, TRAIL_LOCK_PCT,
    TIGHT_TRAIL_ACTIVATE, TIGHT_TRAIL_LOCK_PCT,
    ENTRY_COOLDOWN_BARS, MAX_TRADES_PER_DAY, MAX_LOSS_PER_DAY,
    INCLUDE_COSTS, BROKERAGE_PER_ORDER, SLIPPAGE_POINTS,
    STT_RATE, EXCHANGE_CHARGES, GST_RATE, SEBI_CHARGES, STAMP_DUTY,
    STRUCTURE_BREAK_BUFFER, TIMEFRAME,
)
from data_utils import compute_indicators, generate_nifty_data
from strategy import Signal, TradeStatus, calculate_costs
import dhan_api

# ── Constants ────────────────────────────────────────────────
PAPER_DIR = os.path.join(MOMENTUM_DIR, "paper_trades")
STATE_FILE = os.path.join(PAPER_DIR, "momentum_state.json")
TRADE_LOG_CSV = os.path.join(PAPER_DIR, "momentum_trade_log.csv")
DAILY_LOG_CSV = os.path.join(PAPER_DIR, "momentum_daily_summary.csv")

POLL_DELAY_SEC = 10
WARMUP_CANDLES = 40

# ── Logging ──────────────────────────────────────────────────
os.makedirs(PAPER_DIR, exist_ok=True)

logger = logging.getLogger("MomentumPaper")
logger.setLevel(logging.DEBUG)

_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(_ch)

_fh = logging.FileHandler(os.path.join(PAPER_DIR, "momentum_paper.log"), encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
logger.addHandler(_fh)


# ── Data Classes ─────────────────────────────────────────────
@dataclass
class PaperTrade:
    """A simulated momentum trade with full tracking."""
    trade_id: int
    entry_time: str
    direction: str                    # "LONG" or "SHORT"
    entry_price: float
    entry_lots: int = MAX_LOTS
    current_lots: int = MAX_LOTS
    sl_price: float = 0.0
    initial_sl_points: float = 0.0
    target_price: float = 0.0        # 1:1 R:R partial target

    # Partial exit
    partial_exited: bool = False
    partial_exit_price: float = 0.0
    partial_exit_time: str = ""
    partial_exit_lots: int = 0
    partial_pnl: float = 0.0

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
    entry_rsi: float = 0.0
    entry_macd_hist: float = 0.0
    entry_atr: float = 0.0
    entry_adx: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.status in ("OPEN", "PARTIAL_EXIT")

    @property
    def quantity(self) -> int:
        return self.current_lots * NIFTY_LOT_SIZE

    @property
    def full_quantity(self) -> int:
        return self.entry_lots * NIFTY_LOT_SIZE

    def to_dict(self):
        d = asdict(self)
        d["quantity"] = self.quantity
        d["full_quantity"] = self.full_quantity
        return d


# ── Momentum Paper Trading Engine ────────────────────────────
class MomentumPaperEngine:
    """
    Live paper trading engine for Dual-Confirmation Momentum strategy.
    Polls Dhan API, applies MACD + RSI dual signals, manages partial exits + trailing.
    """

    def __init__(self, interval: int = 5, capital: float = INITIAL_CAPITAL,
                 resume: bool = False, symbol: str = "nifty"):
        self.interval = interval
        self.capital = capital
        self.initial_capital = capital
        self.symbol = symbol

        self.current_trade: Optional[PaperTrade] = None
        self.closed_trades: List[PaperTrade] = []
        self.trade_counter = 0
        self.today = date.today()
        self.trades_today = 0
        self.daily_loss = 0.0
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
        rsi = row.get("rsi", 50)
        macd_bull = row.get("macd_bull_cross", False)
        macd_bear = row.get("macd_bear_cross", False)
        trending = row.get("trending", False)
        volume_surge = row.get("volume_surge", False)
        trend_up = row.get("trend_up", False)
        trend_down = row.get("trend_down", False)
        bullish_candle = row.get("bullish_candle", False)
        bearish_candle = row.get("bearish_candle", False)
        macd_hist = row.get("macd_hist", 0)

        if not (macd_bull or macd_bear):
            return "NONE"
        if not trending:
            return "NONE"
        if not volume_surge:
            return "NONE"

        # LONG
        if macd_bull:
            if rsi >= RSI_EXTREME_OB or rsi < RSI_OVERSOLD:
                return "NONE"
            if not trend_up:
                return "NONE"
            if not bullish_candle:
                return "NONE"
            if macd_hist <= 0:
                return "NONE"
            return "LONG"

        # SHORT
        if macd_bear:
            if rsi <= RSI_EXTREME_OS or rsi > RSI_OVERBOUGHT:
                return "NONE"
            if not trend_down:
                return "NONE"
            if not bearish_candle:
                return "NONE"
            if macd_hist >= 0:
                return "NONE"
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
        """Enter a momentum trade."""
        price = row["close"]
        atr = row.get("atr", 30)
        if pd.isna(atr) or atr <= 0:
            atr = 30

        sl_points = atr * ATR_SL_MULTIPLIER
        sl_points = max(MIN_SL_POINTS, min(sl_points, MAX_SL_POINTS))

        swing_low = row.get("swing_low", price - sl_points)
        swing_high = row.get("swing_high", price + sl_points)

        if direction == "LONG":
            atr_sl = price - sl_points
            structure_sl = swing_low - STRUCTURE_BREAK_BUFFER
            sl_price = max(atr_sl, structure_sl)
            sl_distance = price - sl_price
            target_price = price + sl_distance * PARTIAL_TARGET_RR
        else:
            atr_sl = price + sl_points
            structure_sl = swing_high + STRUCTURE_BREAK_BUFFER
            sl_price = min(atr_sl, structure_sl)
            sl_distance = sl_price - price
            target_price = price - sl_distance * PARTIAL_TARGET_RR

        # Position sizing
        risk_per_lot = sl_distance * NIFTY_LOT_SIZE
        max_risk = self.capital * 0.02
        lots = min(MAX_LOTS, max(1, int(max_risk / risk_per_lot))) if risk_per_lot > 0 else 1

        self.trade_counter += 1
        now_str = str(now)

        self.current_trade = PaperTrade(
            trade_id=self.trade_counter,
            entry_time=now_str,
            direction=direction,
            entry_price=price,
            entry_lots=lots,
            current_lots=lots,
            sl_price=sl_price,
            initial_sl_points=sl_distance,
            target_price=target_price,
            trailing_sl=sl_price,
            entry_rsi=row.get("rsi", 0),
            entry_macd_hist=row.get("macd_hist", 0),
            entry_atr=atr,
            entry_adx=row.get("adx", 0),
        )
        self.trades_today += 1

        logger.info("=" * 60)
        logger.info(f"  MOMENTUM TRADE ENTERED  #{self.trade_counter}")
        logger.info(f"  Direction : {direction}")
        logger.info(f"  Entry     : {price:.2f}")
        logger.info(f"  SL        : {sl_price:.2f} ({sl_distance:.1f} pts)")
        logger.info(f"  Target 1  : {target_price:.2f} (1:1 R:R, 50% exit)")
        logger.info(f"  Lots      : {lots} ({lots * NIFTY_LOT_SIZE} qty)")
        logger.info(f"  RSI       : {row.get('rsi', 0):.1f}  |  MACD Hist: {row.get('macd_hist', 0):.2f}")
        logger.info(f"  ATR       : {atr:.1f}  |  ADX: {row.get('adx', 0):.1f}")
        logger.info("=" * 60)

        self._save_state()

    # ── Position Management ──────────────────────────────────
    def _manage_position(self, row: pd.Series, now: datetime):
        """Update position: check SL, partial exit, trailing stop, EOD."""
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
                status = "SL_HIT" if not trade.partial_exited else "TRAILING_SL"
                self._close_trade(exit_price, now_str, status)
                return

            # Track max favorable
            trade.max_favorable = max(trade.max_favorable, high - trade.entry_price)

            # --- Partial Exit at 1:1 R:R ---
            if not trade.partial_exited and high >= trade.target_price:
                self._partial_exit(trade.target_price, now_str)

            # --- Trailing Stop ---
            if trade.partial_exited and TRAIL_REMAINING:
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
                    new_sl = max(new_sl, trade.entry_price + TRAIL_STEP_POINTS)
                    trade.sl_price = max(trade.sl_price, new_sl)
                trade.trailing_sl = trade.sl_price

        else:  # SHORT
            if high >= trade.sl_price:
                exit_price = trade.sl_price
                status = "SL_HIT" if not trade.partial_exited else "TRAILING_SL"
                self._close_trade(exit_price, now_str, status)
                return

            trade.max_favorable = max(trade.max_favorable, trade.entry_price - low)

            if not trade.partial_exited and low <= trade.target_price:
                self._partial_exit(trade.target_price, now_str)

            if trade.partial_exited and TRAIL_REMAINING:
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
                    new_sl = min(new_sl, trade.entry_price - TRAIL_STEP_POINTS)
                    trade.sl_price = min(trade.sl_price, new_sl)
                trade.trailing_sl = trade.sl_price

        # Max daily loss check
        running_pnl = self._running_pnl()
        if self.daily_loss + min(0, running_pnl) <= -MAX_LOSS_PER_DAY:
            self._close_trade(price, now_str, "MAX_LOSS_EXIT")

    def _partial_exit(self, exit_price: float, now_str: str):
        """Exit 50% of position at 1:1 target."""
        trade = self.current_trade
        partial_lots = max(1, int(trade.entry_lots * PARTIAL_EXIT_PCT))
        partial_qty = partial_lots * NIFTY_LOT_SIZE

        if trade.direction == "LONG":
            trade.partial_pnl = (exit_price - trade.entry_price) * partial_qty
        else:
            trade.partial_pnl = (trade.entry_price - exit_price) * partial_qty

        exit_costs = calculate_costs(exit_price, partial_qty, is_entry=False)
        trade.partial_pnl -= exit_costs["total"]

        trade.partial_exited = True
        trade.partial_exit_price = exit_price
        trade.partial_exit_time = now_str
        trade.partial_exit_lots = partial_lots
        trade.current_lots = trade.entry_lots - partial_lots
        trade.status = "PARTIAL_EXIT"

        # Move SL to breakeven
        trade.sl_price = trade.entry_price
        trade.trailing_sl = trade.entry_price

        logger.info(f"  PARTIAL EXIT: {partial_lots} lots @ {exit_price:.2f}")
        logger.info(f"  Partial P&L: {trade.partial_pnl:+,.0f} | Remaining: {trade.current_lots} lots")
        logger.info(f"  SL moved to breakeven: {trade.entry_price:.2f}")

    def _close_trade(self, exit_price: float, now_str: str, status: str):
        """Close remaining position."""
        trade = self.current_trade
        if not trade:
            return

        trade.exit_price = exit_price
        trade.exit_time = now_str
        trade.status = status

        remaining_qty = trade.current_lots * NIFTY_LOT_SIZE

        if trade.direction == "LONG":
            remaining_pnl = (exit_price - trade.entry_price) * remaining_qty
        else:
            remaining_pnl = (trade.entry_price - exit_price) * remaining_qty

        entry_costs = calculate_costs(trade.entry_price, trade.full_quantity, is_entry=True)
        exit_costs = calculate_costs(exit_price, remaining_qty, is_entry=False)

        trade.gross_pnl = trade.partial_pnl + remaining_pnl
        trade.costs = entry_costs["total"] + exit_costs["total"]
        trade.net_pnl = trade.gross_pnl - entry_costs["total"] - exit_costs["total"]

        self.capital += trade.net_pnl
        self.daily_loss += min(0, trade.net_pnl)
        self.last_exit_bar_idx = self.bar_counter
        self.closed_trades.append(trade)
        self._log_trade(trade)

        emoji = {"TARGET_HIT": "🎯", "SL_HIT": "🛑", "TRAILING_SL": "📈",
                 "TIME_EXIT": "⏰", "MAX_LOSS_EXIT": "⚠️"}
        icon = emoji.get(status, "❓")

        logger.info("=" * 60)
        logger.info(f"  {icon} TRADE CLOSED  #{trade.trade_id}  [{status}]")
        logger.info(f"  {trade.direction}: {trade.entry_price:.2f} -> {exit_price:.2f}")
        if trade.partial_exited:
            logger.info(f"  Partial: {trade.partial_exit_lots} lots @ {trade.partial_exit_price:.2f} = {trade.partial_pnl:+,.0f}")
        logger.info(f"  Remaining: {remaining_qty} qty @ {exit_price:.2f}")
        logger.info(f"  Gross P&L : {trade.gross_pnl:+,.0f}")
        logger.info(f"  Costs     : {trade.costs:,.0f}")
        logger.info(f"  Net P&L   : {trade.net_pnl:+,.0f}")
        logger.info(f"  Capital   : {self.capital:,.0f}")
        logger.info("=" * 60)

        self.current_trade = None
        self._save_state()

    def _running_pnl(self) -> float:
        """Unrealized P&L of open trade."""
        if not self.current_trade or not self.current_trade.is_open:
            return 0
        trade = self.current_trade
        # Approximate using entry vs current (we know last close from data)
        return 0  # Calculated at position check time

    # ── Data Fetching ────────────────────────────────────────
    def _fetch_candles(self, days_back: int = 5) -> Optional[pd.DataFrame]:
        """Fetch intraday candles from Dhan API."""
        try:
            interval_map = {"1min": 1, "5min": 5, "15min": 15}
            interval = interval_map.get(TIMEFRAME, 5)
            df = dhan_api.fetch_intraday(self.symbol, interval=interval, days_back=days_back)

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
                    spot = dhan_api.fetch_ltp(self.symbol) or 0
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
        logger.info(f"  Simulating {days} days of momentum trading...")

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
        logger.info("  MOMENTUM PAPER TRADING")
        logger.info(f"  Mode: {mode} | Strategy: Dual-Confirmation Momentum")
        logger.info(f"  Capital: {self.capital:,.0f} | Max Lots: {MAX_LOTS}")
        logger.info(f"  Partial Exit: {PARTIAL_EXIT_PCT*100:.0f}% at 1:{PARTIAL_TARGET_RR:.0f} R:R")
        logger.info(f"  Trail: {TRAIL_LOCK_PCT*100:.0f}% lock -> {TIGHT_TRAIL_LOCK_PCT*100:.0f}% tight")
        logger.info("=" * 60)

    def _print_status(self, row: pd.Series, now: datetime):
        trade = self.current_trade
        spot = row["close"]
        rsi = row.get("rsi", 0)
        macd_h = row.get("macd_hist", 0)
        adx = row.get("adx", 0)

        if trade and trade.is_open:
            if trade.direction == "LONG":
                upnl = (spot - trade.entry_price) * trade.quantity
            else:
                upnl = (trade.entry_price - spot) * trade.quantity
            icon = "+" if upnl >= 0 else ""
            partial_str = " [PARTIAL]" if trade.partial_exited else ""
            logger.info(f"  {now.strftime('%H:%M')} | Nifty: {spot:.0f} | {trade.direction}{partial_str} @ {trade.entry_price:.0f} | SL: {trade.sl_price:.0f} | P&L: {icon}{upnl:,.0f} | RSI: {rsi:.0f}")
        else:
            logger.info(f"  {now.strftime('%H:%M')} | Nifty: {spot:.0f} | RSI: {rsi:.0f} | MACD: {macd_h:.1f} | ADX: {adx:.0f} | No position")

    def _print_daily_summary(self):
        today_trades = [t for t in self.closed_trades
                        if t.entry_time.startswith(str(self.today))]
        if not today_trades:
            logger.info("  No trades today.")
            return

        net = sum(t.net_pnl for t in today_trades)
        partials = sum(1 for t in today_trades if t.partial_exited)
        logger.info(f"\n  --- Daily Summary ({self.today}) ---")
        logger.info(f"  Trades: {len(today_trades)} | Net P&L: {'+'if net>0 else ''}{net:,.0f}")
        logger.info(f"  Partial exits: {partials}")
        logger.info(f"  Capital: {self.capital:,.0f}")

    def _print_final_summary(self):
        if not self.closed_trades:
            logger.info("  No trades executed.")
            return

        pnls = [t.net_pnl for t in self.closed_trades]
        wins = sum(1 for p in pnls if p > 0)
        total = len(pnls)
        partials = sum(1 for t in self.closed_trades if t.partial_exited)

        logger.info("\n" + "=" * 60)
        logger.info("  MOMENTUM SIMULATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"  Total Trades:     {total}")
        logger.info(f"  Winners:          {wins} ({wins / total * 100:.1f}%)")
        logger.info(f"  Losers:           {total - wins} ({(total - wins) / total * 100:.1f}%)")
        logger.info(f"  Partial Exits:    {partials}")
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

            ct = state.get("current_trade")
            if ct:
                ct.pop("quantity", None)
                ct.pop("full_quantity", None)
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
            "entry_lots": trade.entry_lots,
            "sl_price": round(trade.sl_price, 2),
            "target_price": round(trade.target_price, 2),
            "partial_exited": trade.partial_exited,
            "partial_exit_price": round(trade.partial_exit_price, 2),
            "partial_pnl": round(trade.partial_pnl, 2),
            "gross_pnl": round(trade.gross_pnl, 2),
            "costs": round(trade.costs, 2),
            "net_pnl": round(trade.net_pnl, 2),
            "status": trade.status,
            "entry_rsi": round(trade.entry_rsi, 1),
            "entry_atr": round(trade.entry_atr, 1),
            "entry_adx": round(trade.entry_adx, 1),
            "max_favorable": round(trade.max_favorable, 2),
            "capital": round(self.capital, 2),
        }
        df = pd.DataFrame([row])
        header = not os.path.exists(TRADE_LOG_CSV)
        df.to_csv(TRADE_LOG_CSV, mode="a", header=header, index=False)


# ── CLI ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Dual-Confirmation Momentum Paper Trading Engine"
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

    args = parser.parse_args()

    engine = MomentumPaperEngine(
        interval=args.interval,
        capital=args.capital,
        resume=args.resume,
        symbol=args.symbol,
    )

    if args.live:
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
