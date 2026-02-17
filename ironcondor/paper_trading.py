"""
Iron Condor Paper Trading Engine
==================================
Live paper trading for Bank Nifty Iron Condor strategy.
Polls Dhan API for real-time data, sells OTM call+put spreads with
range-bound filters, manages target/SL, and exits at EOD.

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

# Ensure ironcondor directory imports work
IC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(IC_DIR)
sys.path.insert(0, IC_DIR)
sys.path.insert(0, _ROOT_DIR)

from config import (
    INITIAL_CAPITAL, OPTION_LOT_SIZE, NUM_LOTS,
    ENTRY_TIME, ENTRY_LATEST_TIME, SQUARE_OFF_TIME, TRADING_START, TRADING_END,
    OTM_OFFSET_CE, OTM_OFFSET_PE, WING_WIDTH, STRIKE_ROUNDING, DAYS_TO_EXPIRY,
    SKIP_EXPIRY_DAY,
    TARGET_PCT, SL_BREACH_PCT, COMBINED_SL_MULTIPLIER,
    MAX_LOSS_PER_DAY, DAILY_PROFIT_TARGET,
    VIX_MAX, RSI_LOWER, RSI_UPPER,
    INCLUDE_COSTS, BROKERAGE_PER_ORDER, SLIPPAGE_POINTS,
    STT_RATE, EXCHANGE_CHARGES, GST_RATE, SEBI_CHARGES, STAMP_DUTY,
    BANKNIFTY_IV, RISK_FREE_RATE, TIMEFRAME,
)
from data_utils import (
    black_scholes_price, get_atm_strike, get_iron_condor_strikes,
    compute_rsi, generate_banknifty_data,
)
from strategy import calculate_costs, LegStatus
import dhan_api

# ── Constants ────────────────────────────────────────────────
PAPER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_trades")
STATE_FILE = os.path.join(PAPER_DIR, "ic_state.json")
TRADE_LOG_CSV = os.path.join(PAPER_DIR, "ic_trade_log.csv")
DAILY_LOG_CSV = os.path.join(PAPER_DIR, "ic_daily_summary.csv")

POLL_DELAY_SEC = 10
WARMUP_CANDLES = 30

# ── Logging ──────────────────────────────────────────────────
os.makedirs(PAPER_DIR, exist_ok=True)

logger = logging.getLogger("IronCondorPaper")
logger.setLevel(logging.DEBUG)

_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(_ch)

_fh = logging.FileHandler(os.path.join(PAPER_DIR, "ic_paper.log"), encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
logger.addHandler(_fh)


# ── Data Classes ─────────────────────────────────────────────
@dataclass
class PaperLeg:
    """One leg of the paper Iron Condor."""
    option_type: str          # "CE" or "PE"
    leg_role: str             # "SHORT" or "LONG"
    strike: float
    entry_premium: float
    current_premium: float = 0.0
    exit_premium: float = 0.0
    status: str = "OPEN"

    def to_dict(self):
        return asdict(self)


@dataclass
class PaperPosition:
    """Current Iron Condor paper position."""
    date: str
    entry_spot: float
    entry_time: str

    short_ce: Optional[PaperLeg] = None
    long_ce: Optional[PaperLeg] = None
    short_pe: Optional[PaperLeg] = None
    long_pe: Optional[PaperLeg] = None

    net_credit: float = 0.0
    current_pnl: float = 0.0
    exit_reason: str = ""

    vix_at_entry: float = 0.0
    rsi_at_entry: float = 0.0

    @property
    def is_open(self):
        if self.short_ce and self.short_ce.status == "OPEN":
            return True
        return False

    def to_dict(self):
        d = {
            "date": self.date,
            "entry_spot": self.entry_spot,
            "entry_time": self.entry_time,
            "net_credit": self.net_credit,
            "current_pnl": self.current_pnl,
            "exit_reason": self.exit_reason,
            "vix_at_entry": self.vix_at_entry,
            "rsi_at_entry": self.rsi_at_entry,
        }
        if self.short_ce:
            d["short_ce"] = self.short_ce.to_dict()
        if self.long_ce:
            d["long_ce"] = self.long_ce.to_dict()
        if self.short_pe:
            d["short_pe"] = self.short_pe.to_dict()
        if self.long_pe:
            d["long_pe"] = self.long_pe.to_dict()
        return d


@dataclass
class PaperState:
    """Persistent paper trading state."""
    capital: float = INITIAL_CAPITAL
    initial_capital: float = INITIAL_CAPITAL
    today_pnl: float = 0.0
    total_trades: int = 0
    total_pnl: float = 0.0
    wins: int = 0
    losses: int = 0
    current_position: Optional[dict] = None
    last_update: str = ""

    def to_dict(self):
        return asdict(self)


# ── State Persistence ────────────────────────────────────────
def save_state(state: PaperState):
    with open(STATE_FILE, "w") as f:
        json.dump(state.to_dict(), f, indent=2, default=str)


def load_state() -> PaperState:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                data = json.load(f)
            return PaperState(**{k: v for k, v in data.items()
                                if k in PaperState.__dataclass_fields__})
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")
    return PaperState()


def append_trade(trade_data: dict):
    """Append a completed trade to the CSV log."""
    df = pd.DataFrame([trade_data])
    header = not os.path.exists(TRADE_LOG_CSV) or os.path.getsize(TRADE_LOG_CSV) == 0
    df.to_csv(TRADE_LOG_CSV, mode="a", header=header, index=False)


def append_daily(daily_data: dict):
    """Append daily summary row."""
    df = pd.DataFrame([daily_data])
    header = not os.path.exists(DAILY_LOG_CSV) or os.path.getsize(DAILY_LOG_CSV) == 0
    df.to_csv(DAILY_LOG_CSV, mode="a", header=header, index=False)


# ── Paper Trading Engine ─────────────────────────────────────
class IronCondorPaperTrader:
    """Manages Iron Condor paper trading lifecycle."""

    def __init__(self, state: PaperState, symbol: str = "banknifty"):
        self.state = state
        self.symbol = symbol
        self.quantity = OPTION_LOT_SIZE * NUM_LOTS
        self.position: Optional[PaperPosition] = None
        self._running = True

        sig.signal(sig.SIGINT, self._handle_signal)
        sig.signal(sig.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        logger.info("Shutdown signal received.")
        self._running = False

    def run_live(self, resume: bool = False):
        """Main live paper trading loop."""
        logger.info("=" * 60)
        logger.info("  IRON CONDOR PAPER TRADER — LIVE MODE")
        logger.info(f"  Capital: ₹{self.state.capital:,.0f}")
        logger.info(f"  CE Offset: {OTM_OFFSET_CE} | PE Offset: {OTM_OFFSET_PE} | Wings: {WING_WIDTH}")
        logger.info("=" * 60)

        if resume and self.state.current_position:
            logger.info("  Resuming from saved position...")

        while self._running:
            now = datetime.now()

            # Only trade on weekdays
            if now.weekday() >= 5:
                logger.info("  Weekend — sleeping 1 hour")
                time.sleep(3600)
                continue

            # Skip expiry day (Thursday)
            if SKIP_EXPIRY_DAY and now.weekday() == 3:
                logger.info("  Expiry day (Thursday) — skipping")
                time.sleep(3600)
                continue

            # Before market
            if now.hour < 9 or (now.hour == 9 and now.minute < 15):
                logger.info(f"  Pre-market — waiting until 09:15")
                time.sleep(60)
                continue

            # After market close
            if now.hour >= 15 and now.minute >= 35:
                logger.info("  Market closed — running EOD tasks")
                self._eod_tasks()
                time.sleep(3600)
                continue

            # ── Fetch current spot ──
            try:
                spot = dhan_api.fetch_ltp(self.symbol)
                if spot is None or spot <= 0:
                    time.sleep(POLL_DELAY_SEC)
                    continue
            except Exception as e:
                logger.error(f"  Fetch error: {e}")
                time.sleep(POLL_DELAY_SEC)
                continue

            # ── Entry logic ──
            entry_h, entry_m = map(int, ENTRY_TIME.split(":"))
            if self.position is None or not self.position.is_open:
                if now.hour >= entry_h and now.minute >= entry_m:
                    latest_h, latest_m = map(int, ENTRY_LATEST_TIME.split(":"))
                    if now.hour < latest_h or (now.hour == latest_h and now.minute <= latest_m):
                        self._try_entry(spot, now)

            # ── Position management ──
            if self.position and self.position.is_open:
                self._manage_position(spot, now)

            # Save state periodically
            self.state.last_update = now.strftime("%Y-%m-%d %H:%M:%S")
            save_state(self.state)

            time.sleep(POLL_DELAY_SEC)

    def _try_entry(self, spot: float, now: datetime):
        """Attempt to enter an Iron Condor position."""
        # TODO: Add VIX and RSI filter checks with live data
        logger.info(f"  Evaluating entry at spot={spot:.0f}")

        # Get strikes
        sell_ce, buy_ce, sell_pe, buy_pe = get_iron_condor_strikes(
            spot, OTM_OFFSET_CE, OTM_OFFSET_PE, WING_WIDTH, STRIKE_ROUNDING
        )

        iv = BANKNIFTY_IV / 100.0
        r = RISK_FREE_RATE / 100.0
        dte = DAYS_TO_EXPIRY / 365.0

        sell_ce_prem = max(black_scholes_price(spot, sell_ce, dte, iv, r, "CE"), 3.0)
        buy_ce_prem = max(black_scholes_price(spot, buy_ce, dte, iv, r, "CE"), 0.5)
        sell_pe_prem = max(black_scholes_price(spot, sell_pe, dte, iv, r, "PE"), 3.0)
        buy_pe_prem = max(black_scholes_price(spot, buy_pe, dte, iv, r, "PE"), 0.5)

        net_credit = (sell_ce_prem - buy_ce_prem) + (sell_pe_prem - buy_pe_prem)
        if net_credit < 5.0:
            logger.info(f"  Credit too low ({net_credit:.1f} pts) — skipping")
            return

        self.position = PaperPosition(
            date=str(now.date()),
            entry_spot=spot,
            entry_time=now.strftime("%H:%M:%S"),
            short_ce=PaperLeg("CE", "SHORT", sell_ce, sell_ce_prem, sell_ce_prem),
            long_ce=PaperLeg("CE", "LONG", buy_ce, buy_ce_prem, buy_ce_prem),
            short_pe=PaperLeg("PE", "SHORT", sell_pe, sell_pe_prem, sell_pe_prem),
            long_pe=PaperLeg("PE", "LONG", buy_pe, buy_pe_prem, buy_pe_prem),
            net_credit=net_credit,
        )

        logger.info(f"  ENTRY: Iron Condor")
        logger.info(f"    Sell {sell_ce}CE @ {sell_ce_prem:.1f} | Buy {buy_ce}CE @ {buy_ce_prem:.1f}")
        logger.info(f"    Sell {sell_pe}PE @ {sell_pe_prem:.1f} | Buy {buy_pe}PE @ {buy_pe_prem:.1f}")
        logger.info(f"    Net Credit: {net_credit:.1f} pts | Total: ₹{net_credit * self.quantity:,.0f}")

        self.state.current_position = self.position.to_dict()

    def _manage_position(self, spot: float, now: datetime):
        """Monitor and manage open Iron Condor position."""
        pos = self.position
        iv = BANKNIFTY_IV / 100.0
        r = RISK_FREE_RATE / 100.0
        dte = max(DAYS_TO_EXPIRY - 0.5, 0.01) / 365.0  # Approximate remaining DTE

        # Update premiums
        pos.short_ce.current_premium = black_scholes_price(spot, pos.short_ce.strike, dte, iv, r, "CE")
        pos.long_ce.current_premium = black_scholes_price(spot, pos.long_ce.strike, dte, iv, r, "CE")
        pos.short_pe.current_premium = black_scholes_price(spot, pos.short_pe.strike, dte, iv, r, "PE")
        pos.long_pe.current_premium = black_scholes_price(spot, pos.long_pe.strike, dte, iv, r, "PE")

        current_spread = (
            (pos.short_ce.current_premium - pos.long_ce.current_premium) +
            (pos.short_pe.current_premium - pos.long_pe.current_premium)
        )
        credit_captured = pos.net_credit - current_spread
        pnl = credit_captured * self.quantity
        pos.current_pnl = round(pnl, 2)

        # Check target
        if credit_captured >= pos.net_credit * TARGET_PCT:
            self._close_position("Target Hit", now)
            return

        # Check breach
        breach_dist = WING_WIDTH * SL_BREACH_PCT
        if spot >= pos.short_ce.strike + breach_dist:
            self._close_position("CE Breached", now)
            return
        if spot <= pos.short_pe.strike - breach_dist:
            self._close_position("PE Breached", now)
            return

        # Check combined SL
        if current_spread >= pos.net_credit * COMBINED_SL_MULTIPLIER:
            self._close_position("Combined SL", now)
            return

        # Check max loss
        if pnl < -MAX_LOSS_PER_DAY:
            self._close_position("Max Loss", now)
            return

        # EOD exit
        exit_h, exit_m = map(int, SQUARE_OFF_TIME.split(":"))
        if now.hour >= exit_h and now.minute >= exit_m:
            self._close_position("EOD Square-off", now)
            return

        logger.debug(f"  Spot={spot:.0f} | Spread={current_spread:.1f} | P&L=₹{pnl:+,.0f}")

    def _close_position(self, reason: str, now: datetime):
        """Close all legs and log the trade."""
        pos = self.position
        for leg in [pos.short_ce, pos.long_ce, pos.short_pe, pos.long_pe]:
            if leg:
                leg.exit_premium = leg.current_premium
                leg.status = reason

        pos.exit_reason = reason

        # Calculate P&L
        gross = 0
        for leg in [pos.short_ce, pos.long_ce, pos.short_pe, pos.long_pe]:
            if leg:
                if leg.leg_role == "SHORT":
                    gross += (leg.entry_premium - leg.exit_premium) * self.quantity
                else:
                    gross += (leg.exit_premium - leg.entry_premium) * self.quantity

        costs = sum(
            calculate_costs(leg.entry_premium, self.quantity)["total"] +
            calculate_costs(leg.exit_premium, self.quantity, is_entry=False)["total"]
            for leg in [pos.short_ce, pos.long_ce, pos.short_pe, pos.long_pe] if leg
        )
        net = gross - costs

        logger.info(f"  EXIT: {reason}")
        logger.info(f"    Gross: ₹{gross:+,.0f} | Costs: ₹{costs:,.0f} | Net: ₹{net:+,.0f}")

        # Update state
        self.state.total_trades += 1
        self.state.today_pnl += net
        self.state.total_pnl += net
        self.state.capital += net
        if net > 0:
            self.state.wins += 1
        else:
            self.state.losses += 1
        self.state.current_position = None

        # Log trade
        append_trade({
            "trade_id": self.state.total_trades,
            "date": pos.date,
            "entry_spot": pos.entry_spot,
            "exit_spot": 0,
            "sell_ce_strike": pos.short_ce.strike if pos.short_ce else 0,
            "buy_ce_strike": pos.long_ce.strike if pos.long_ce else 0,
            "sell_pe_strike": pos.short_pe.strike if pos.short_pe else 0,
            "buy_pe_strike": pos.long_pe.strike if pos.long_pe else 0,
            "net_credit": round(pos.net_credit, 2),
            "gross_pnl": round(gross, 2),
            "costs": round(costs, 2),
            "net_pnl": round(net, 2),
            "exit_reason": reason,
            "capital": round(self.state.capital, 2),
        })

        save_state(self.state)
        self.position = None

    def _eod_tasks(self):
        """End-of-day cleanup."""
        if self.position and self.position.is_open:
            self._close_position("EOD Square-off", datetime.now())

        if self.state.today_pnl != 0:
            append_daily({
                "date": str(date.today()),
                "day_pnl": round(self.state.today_pnl, 2),
                "capital": round(self.state.capital, 2),
                "trades": self.state.total_trades,
                "wins": self.state.wins,
                "losses": self.state.losses,
            })

        logger.info(f"  EOD: Today P&L = ₹{self.state.today_pnl:+,.0f}")
        logger.info(f"  Capital: ₹{self.state.capital:,.0f}")
        self.state.today_pnl = 0.0
        save_state(self.state)

    def run_simulate(self, days: int = 5):
        """Simulate paper trading using recent generated data."""
        logger.info(f"  Simulating {days} days of Iron Condor paper trading...")

        df = generate_banknifty_data(days=days, timeframe=TIMEFRAME)
        logger.info(f"  Data: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")

        from strategy import IronCondorBacktest
        engine = IronCondorBacktest()
        engine.capital = self.state.capital
        result = engine.run(df)

        for t in result.trades:
            net = t.net_pnl
            self.state.total_trades += 1
            self.state.total_pnl += net
            self.state.capital += net
            if net > 0:
                self.state.wins += 1
            else:
                self.state.losses += 1

            append_trade({
                "trade_id": self.state.total_trades,
                "date": str(t.date),
                "entry_spot": round(t.entry_spot, 2),
                "exit_spot": round(t.spot_at_exit, 2),
                "sell_ce_strike": t.short_ce.strike if t.short_ce else 0,
                "buy_ce_strike": t.long_ce.strike if t.long_ce else 0,
                "sell_pe_strike": t.short_pe.strike if t.short_pe else 0,
                "buy_pe_strike": t.long_pe.strike if t.long_pe else 0,
                "net_credit": round(t.total_net_credit, 2),
                "gross_pnl": round(t.gross_pnl, 2),
                "costs": round(t.total_costs, 2),
                "net_pnl": round(net, 2),
                "exit_reason": t.exit_reason,
                "capital": round(self.state.capital, 2),
            })

        save_state(self.state)
        engine.print_summary(result)
        logger.info(f"\n  Simulation complete. Capital: ₹{self.state.capital:,.0f}")


# ── Main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Iron Condor Paper Trader")
    parser.add_argument("--live", action="store_true", help="Run live paper trading")
    parser.add_argument("--simulate", action="store_true", help="Simulate on sample data")
    parser.add_argument("--days", type=int, default=5, help="Simulation days (default: 5)")
    parser.add_argument("--resume", action="store_true", help="Resume from saved state")
    parser.add_argument("--reset", action="store_true", help="Reset state and start fresh")
    parser.add_argument("--symbol", type=str, default="banknifty",
                        choices=["nifty", "banknifty", "finnifty", "midcapnifty", "sensex"],
                        help="Index to trade (default: banknifty)")
    args = parser.parse_args()

    if args.reset:
        state = PaperState()
        save_state(state)
        logger.info("  State reset to default.")
        return

    state = load_state() if args.resume or args.live else PaperState()
    trader = IronCondorPaperTrader(state, symbol=args.symbol)

    if args.live:
        trader.run_live(resume=args.resume)
    elif args.simulate:
        trader.run_simulate(days=args.days)
    else:
        print("  Usage: python paper_trading.py --live  OR  --simulate --days 5")
        print("  Add --resume to continue from saved state")


if __name__ == "__main__":
    main()
