"""
Iron Condor Strategy Engine
=============================
Sells OTM Call Spread + OTM Put Spread on Bank Nifty weekly options.
4-leg credit spread with defined risk. Filters for range-bound conditions.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple
from datetime import datetime, time

from config import (
    INITIAL_CAPITAL, OPTION_LOT_SIZE, NUM_LOTS,
    ENTRY_TIME, ENTRY_LATEST_TIME, SQUARE_OFF_TIME, TRADING_START, TRADING_END,
    OTM_OFFSET_CE, OTM_OFFSET_PE, WING_WIDTH, STRIKE_ROUNDING, DAYS_TO_EXPIRY,
    SKIP_EXPIRY_DAY,
    TARGET_PCT, MAX_TARGET_PCT, SL_BREACH_PCT,
    COMBINED_SL_MULTIPLIER, MAX_LOSS_PER_DAY, DAILY_PROFIT_TARGET,
    VIX_MAX, RSI_LOWER, RSI_UPPER, RSI_PERIOD, BB_PERIOD, BB_STD, BB_WIDTH_MAX,
    INCLUDE_COSTS, BROKERAGE_PER_ORDER, SLIPPAGE_POINTS,
    STT_RATE, EXCHANGE_CHARGES, GST_RATE, SEBI_CHARGES, STAMP_DUTY,
    BANKNIFTY_IV, RISK_FREE_RATE,
)
from data_utils import (
    black_scholes_price, bs_greeks, get_atm_strike, get_iron_condor_strikes,
    compute_rsi, compute_bollinger_bands, compute_atr,
)


# ─── Enums & Data Classes ───────────────────────────────────────────────────

class LegStatus(Enum):
    OPEN = "OPEN"
    EXITED_TARGET = "EXITED_TARGET"
    EXITED_SL = "EXITED_SL"
    EXITED_BREACH = "EXITED_BREACH"
    EXITED_EOD = "EXITED_EOD"
    EXITED_MAXLOSS = "EXITED_MAXLOSS"
    EXITED_COMBINED = "EXITED_COMBINED"


@dataclass
class Leg:
    """Single leg of the Iron Condor (CE or PE, short or long)."""
    option_type: str               # "CE" or "PE"
    leg_role: str                  # "SHORT" or "LONG" (protection)
    strike: float
    entry_premium: float
    current_premium: float = 0.0
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    exit_premium: float = 0.0
    status: LegStatus = LegStatus.OPEN

    # P&L
    gross_pnl: float = 0.0
    costs: float = 0.0
    net_pnl: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.status == LegStatus.OPEN


@dataclass
class IronCondorTrade:
    """One day's Iron Condor trade (4 legs)."""
    date: datetime
    entry_spot: float

    # Short legs (sold)
    short_ce: Optional[Leg] = None   # Sell CE (higher OTM)
    long_ce: Optional[Leg] = None    # Buy CE (protection — further OTM)
    short_pe: Optional[Leg] = None   # Sell PE (lower OTM)
    long_pe: Optional[Leg] = None    # Buy PE (protection — further OTM)

    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None

    # Credit collected
    net_credit_ce: float = 0.0       # Sell CE prem - Buy CE prem
    net_credit_pe: float = 0.0       # Sell PE prem - Buy PE prem
    total_net_credit: float = 0.0    # Total credit collected
    max_risk_per_lot: float = 0.0    # Wing width - net credit

    # P&L
    gross_pnl: float = 0.0
    total_costs: float = 0.0
    net_pnl: float = 0.0

    # Exit
    exit_reason: str = ""
    spot_at_exit: float = 0.0

    # Filters at entry
    vix_at_entry: float = 0.0
    rsi_at_entry: float = 0.0
    bb_width_at_entry: float = 0.0
    skipped_reason: str = ""

    @property
    def is_winner(self) -> bool:
        return self.net_pnl > 0

    @property
    def credit_captured_pct(self) -> float:
        """How much of the initial credit was captured as profit."""
        if self.total_net_credit <= 0:
            return 0
        return max(0, self.net_pnl / (self.total_net_credit * self.quantity)) * 100 if hasattr(self, 'quantity') else 0


@dataclass
class BacktestResult:
    """Aggregated backtest results."""
    trades: List[IronCondorTrade]
    equity_curve: List[float]
    daily_pnl: List[dict]
    initial_capital: float
    final_capital: float
    skipped_days: int = 0

    # Summary stats
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    win_rate: float = 0.0
    total_gross_pnl: float = 0.0
    total_costs: float = 0.0
    total_net_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_inr: float = 0.0
    roi_pct: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    best_month: float = 0.0
    worst_month: float = 0.0
    avg_monthly_return: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    avg_credit_collected: float = 0.0
    avg_credit_captured_pct: float = 0.0


# ─── Transaction Cost Calculator ────────────────────────────────────────────

def calculate_costs(premium: float, quantity: int, is_entry: bool = True) -> dict:
    """Calculate all transaction costs for one leg."""
    if not INCLUDE_COSTS:
        return {"total": 0, "brokerage": 0, "stt": 0, "exchange": 0,
                "gst": 0, "sebi": 0, "stamp": 0, "slippage": 0}

    turnover = premium * quantity
    brokerage = min(BROKERAGE_PER_ORDER, turnover * 0.0003)
    stt = turnover * STT_RATE if not is_entry else 0
    exchange = turnover * EXCHANGE_CHARGES
    gst = (brokerage + exchange) * GST_RATE
    sebi = turnover * SEBI_CHARGES
    stamp = turnover * STAMP_DUTY if is_entry else 0
    slippage = SLIPPAGE_POINTS * quantity

    total = brokerage + stt + exchange + gst + sebi + stamp + slippage
    return {
        "total": round(total, 2),
        "brokerage": round(brokerage, 2),
        "stt": round(stt, 2),
        "exchange": round(exchange, 2),
        "gst": round(gst, 2),
        "sebi": round(sebi, 2),
        "stamp": round(stamp, 2),
        "slippage": round(slippage, 2),
    }


# ─── Core Iron Condor Backtest Engine ───────────────────────────────────────

class IronCondorBacktest:
    """
    Bank Nifty Iron Condor Backtest Engine.

    Logic:
    1. After 10:05 AM, check range-bound filters (VIX, RSI, Bollinger Bands)
    2. If filters pass, sell OTM Call Spread + OTM Put Spread (4 legs)
    3. Target: Close at 50% of credit captured
    4. SL: Exit if spot breaches sold strikes by 25% of wing width
    5. Also exit if combined spread value rises to 2x net credit
    6. Square off at 15:20 or on target/SL
    7. Skip expiry days (Thursday)
    """

    def __init__(self):
        self.capital = INITIAL_CAPITAL
        self.lot_size = OPTION_LOT_SIZE
        self.quantity = OPTION_LOT_SIZE * NUM_LOTS
        self.trades: List[IronCondorTrade] = []
        self.equity_curve: List[float] = [INITIAL_CAPITAL]
        self.daily_pnl: List[dict] = []
        self.skipped_days = 0

    def run(self, df: pd.DataFrame) -> BacktestResult:
        """Run backtest on 5-min Bank Nifty OHLCV data."""
        print("\n" + "=" * 70)
        print("  IRON CONDOR — BANK NIFTY WEEKLY OPTIONS BACKTEST")
        print("=" * 70)
        print(f"  Capital: ₹{INITIAL_CAPITAL:,.0f}")
        print(f"  Lots: {NUM_LOTS} × {OPTION_LOT_SIZE} = {self.quantity} qty per side")
        print(f"  CE Offset: {OTM_OFFSET_CE} pts | PE Offset: {OTM_OFFSET_PE} pts | Wings: {WING_WIDTH} pts")
        print(f"  Target: {TARGET_PCT*100:.0f}% credit capture | SL: {SL_BREACH_PCT*100:.0f}% breach")
        print(f"  Filters: VIX < {VIX_MAX}, RSI {RSI_LOWER}-{RSI_UPPER}, BB width < {BB_WIDTH_MAX}")
        print("=" * 70)

        # Precompute daily RSI + BB on close prices
        df = df.copy()
        df["date"] = df.index.date
        df["rsi"] = compute_rsi(df["close"], RSI_PERIOD)
        bb = compute_bollinger_bands(df["close"], BB_PERIOD, BB_STD)
        df = pd.concat([df, bb], axis=1)

        trading_days = df.groupby("date")
        day_count = 0

        for date_val, day_data in trading_days:
            day_count += 1
            if len(day_data) < 10:
                continue

            trade = self._process_day(date_val, day_data)
            if trade is not None:
                if trade.skipped_reason:
                    self.skipped_days += 1
                    continue

                self.trades.append(trade)
                self.capital += trade.net_pnl
                self.equity_curve.append(self.capital)

                self.daily_pnl.append({
                    "date": date_val,
                    "gross_pnl": trade.gross_pnl,
                    "costs": trade.total_costs,
                    "net_pnl": trade.net_pnl,
                    "capital": self.capital,
                    "exit_reason": trade.exit_reason,
                    "entry_spot": trade.entry_spot,
                    "exit_spot": trade.spot_at_exit,
                    "net_credit": trade.total_net_credit,
                    "vix": trade.vix_at_entry,
                    "rsi": trade.rsi_at_entry,
                    "sell_ce": trade.short_ce.strike if trade.short_ce else 0,
                    "sell_pe": trade.short_pe.strike if trade.short_pe else 0,
                })

        print(f"\n  Total trading days: {day_count}")
        print(f"  Skipped (filters): {self.skipped_days}")
        print(f"  Trades taken: {len(self.trades)}")

        return self._compute_results()

    def _process_day(self, date_val, day_data: pd.DataFrame) -> Optional[IronCondorTrade]:
        """Process one trading day."""

        # Check if it's expiry day (Thursday) — skip
        sample_ts = day_data.index[0]
        if SKIP_EXPIRY_DAY and sample_ts.weekday() == 3:
            trade = IronCondorTrade(date=date_val, entry_spot=0)
            trade.skipped_reason = "Expiry day (Thursday)"
            return trade

        # Find entry bar (10:05)
        entry_hour, entry_min = map(int, ENTRY_TIME.split(":"))
        latest_hour, latest_min = map(int, ENTRY_LATEST_TIME.split(":"))
        exit_hour, exit_min = map(int, SQUARE_OFF_TIME.split(":"))

        entry_bar = None
        for ts in day_data.index:
            if (ts.hour == entry_hour and ts.minute >= entry_min) or ts.hour > entry_hour:
                if ts.hour < latest_hour or (ts.hour == latest_hour and ts.minute <= latest_min):
                    entry_bar = ts
                    break

        if entry_bar is None:
            return None

        entry_spot = day_data.loc[entry_bar, "close"]

        # ── Range-bound filters ──
        vix_today = day_data["vix"].iloc[0] if "vix" in day_data.columns else 14.0
        rsi_at_entry = day_data.loc[entry_bar, "rsi"] if "rsi" in day_data.columns else 50.0
        bb_width = day_data.loc[entry_bar, "bb_width"] if "bb_width" in day_data.columns else 0.02

        # VIX filter
        if vix_today > VIX_MAX:
            trade = IronCondorTrade(date=date_val, entry_spot=entry_spot)
            trade.skipped_reason = f"VIX too high ({vix_today:.1f} > {VIX_MAX})"
            trade.vix_at_entry = vix_today
            return trade

        # RSI filter: must be in range-bound zone (40-60)
        if rsi_at_entry < RSI_LOWER or rsi_at_entry > RSI_UPPER:
            trade = IronCondorTrade(date=date_val, entry_spot=entry_spot)
            trade.skipped_reason = f"RSI out of range ({rsi_at_entry:.1f})"
            trade.rsi_at_entry = rsi_at_entry
            return trade

        # Bollinger Band width filter: must be tight
        if bb_width > BB_WIDTH_MAX:
            trade = IronCondorTrade(date=date_val, entry_spot=entry_spot)
            trade.skipped_reason = f"BB too wide ({bb_width:.4f} > {BB_WIDTH_MAX})"
            trade.bb_width_at_entry = bb_width
            return trade

        # ── Strike selection ──
        iv_today = day_data["daily_iv"].iloc[0] / 100.0 if "daily_iv" in day_data.columns else BANKNIFTY_IV / 100.0
        r = RISK_FREE_RATE / 100.0

        sell_ce_strike, buy_ce_strike, sell_pe_strike, buy_pe_strike = get_iron_condor_strikes(
            entry_spot, OTM_OFFSET_CE, OTM_OFFSET_PE, WING_WIDTH, STRIKE_ROUNDING
        )

        dte = DAYS_TO_EXPIRY / 365.0

        # Compute entry premiums (4 legs)
        sell_ce_prem = black_scholes_price(entry_spot, sell_ce_strike, dte, iv_today, r, "CE")
        buy_ce_prem = black_scholes_price(entry_spot, buy_ce_strike, dte, iv_today, r, "CE")
        sell_pe_prem = black_scholes_price(entry_spot, sell_pe_strike, dte, iv_today, r, "PE")
        buy_pe_prem = black_scholes_price(entry_spot, buy_pe_strike, dte, iv_today, r, "PE")

        # Ensure minimum premiums
        sell_ce_prem = max(sell_ce_prem, 3.0)
        sell_pe_prem = max(sell_pe_prem, 3.0)
        buy_ce_prem = max(buy_ce_prem, 0.5)
        buy_pe_prem = max(buy_pe_prem, 0.5)

        net_credit_ce = sell_ce_prem - buy_ce_prem
        net_credit_pe = sell_pe_prem - buy_pe_prem
        total_net_credit = net_credit_ce + net_credit_pe

        # Skip if credit is too low (not worth the risk)
        if total_net_credit < 5.0:
            trade = IronCondorTrade(date=date_val, entry_spot=entry_spot)
            trade.skipped_reason = f"Credit too low ({total_net_credit:.1f} pts)"
            return trade

        max_risk_per_lot = WING_WIDTH - total_net_credit

        # Create 4 legs
        short_ce = Leg("CE", "SHORT", sell_ce_strike, sell_ce_prem, sell_ce_prem, entry_bar)
        long_ce = Leg("CE", "LONG", buy_ce_strike, buy_ce_prem, buy_ce_prem, entry_bar)
        short_pe = Leg("PE", "SHORT", sell_pe_strike, sell_pe_prem, sell_pe_prem, entry_bar)
        long_pe = Leg("PE", "LONG", buy_pe_strike, buy_pe_prem, buy_pe_prem, entry_bar)

        trade = IronCondorTrade(
            date=date_val,
            entry_spot=entry_spot,
            short_ce=short_ce,
            long_ce=long_ce,
            short_pe=short_pe,
            long_pe=long_pe,
            entry_time=entry_bar,
            net_credit_ce=net_credit_ce,
            net_credit_pe=net_credit_pe,
            total_net_credit=total_net_credit,
            max_risk_per_lot=max_risk_per_lot,
            vix_at_entry=vix_today,
            rsi_at_entry=rsi_at_entry,
            bb_width_at_entry=bb_width if not np.isnan(bb_width) else 0,
        )

        # Entry costs (4 legs)
        entry_costs = sum(
            calculate_costs(leg.entry_premium, self.quantity, is_entry=True)["total"]
            for leg in [short_ce, long_ce, short_pe, long_pe]
        )

        # ── Process bars after entry ──
        bars_after_entry = day_data.loc[entry_bar:]

        for ts in bars_after_entry.index:
            if ts == entry_bar:
                continue

            spot = day_data.loc[ts, "close"]
            high = day_data.loc[ts, "high"]
            low = day_data.loc[ts, "low"]

            # Time decay within the day
            minutes_from_open = (ts.hour - 9) * 60 + (ts.minute - 15)
            day_fraction = max(0, min(minutes_from_open / 375, 1.0))
            remaining_dte = max(DAYS_TO_EXPIRY - day_fraction, 0.01) / 365.0

            # IV adjustment on spot moves
            spot_move_pct = abs(spot - trade.entry_spot) / trade.entry_spot * 100
            iv_adj = iv_today
            if spot_move_pct > 1.0:
                iv_adj *= (1 + spot_move_pct * 0.3)
            elif spot_move_pct > 0.5:
                iv_adj *= (1 + spot_move_pct * 0.15)
            iv_adj *= (1 + np.random.normal(0, 0.03))

            # Update all 4 leg premiums
            if short_ce.is_open:
                short_ce.current_premium = black_scholes_price(spot, sell_ce_strike, remaining_dte, iv_adj, r, "CE")
            if long_ce.is_open:
                long_ce.current_premium = black_scholes_price(spot, buy_ce_strike, remaining_dte, iv_adj, r, "CE")
            if short_pe.is_open:
                short_pe.current_premium = black_scholes_price(spot, sell_pe_strike, remaining_dte, iv_adj, r, "PE")
            if long_pe.is_open:
                long_pe.current_premium = black_scholes_price(spot, buy_pe_strike, remaining_dte, iv_adj, r, "PE")

            # Current spread values
            current_ce_spread = short_ce.current_premium - long_ce.current_premium
            current_pe_spread = short_pe.current_premium - long_pe.current_premium
            current_total_spread = current_ce_spread + current_pe_spread

            # Credit captured so far (positive = profit for us)
            credit_captured = total_net_credit - current_total_spread

            # ── Check Target: 50% credit capture ──
            if credit_captured >= total_net_credit * TARGET_PCT:
                self._close_all_legs(trade, ts, "Target Hit (50% credit)")
                break

            # ── Check Breach SL: spot breaches sold strike ──
            breach_distance = WING_WIDTH * SL_BREACH_PCT

            # Call side breach
            if high >= sell_ce_strike + breach_distance:
                self._close_all_legs(trade, ts, "CE Strike Breached")
                break

            # Put side breach
            if low <= sell_pe_strike - breach_distance:
                self._close_all_legs(trade, ts, "PE Strike Breached")
                break

            # ── Check Combined SL: spread rising to 2x credit ──
            if current_total_spread >= total_net_credit * COMBINED_SL_MULTIPLIER:
                self._close_all_legs(trade, ts, "Combined SL (2x credit)")
                break

            # ── Max Daily Loss ──
            running_pnl = (total_net_credit - current_total_spread) * self.quantity
            if running_pnl < -MAX_LOSS_PER_DAY:
                self._close_all_legs(trade, ts, "Max Daily Loss")
                break

            # ── EOD Exit ──
            if ts.hour == exit_hour and ts.minute >= exit_min:
                self._close_all_legs(trade, ts, "EOD Square-off")
                break
            elif ts.hour > exit_hour:
                self._close_all_legs(trade, ts, "EOD Square-off")
                break

        # Force close any remaining open legs
        all_legs = [short_ce, long_ce, short_pe, long_pe]
        if any(leg.is_open for leg in all_legs):
            self._close_all_legs(trade, day_data.index[-1], "EOD Square-off")

        # ── Calculate P&L ──
        # Short legs: P&L = (entry - exit) × qty (profit when premium decays)
        # Long legs: P&L = (exit - entry) × qty (hedge cost)
        short_ce.gross_pnl = (short_ce.entry_premium - short_ce.exit_premium) * self.quantity
        long_ce.gross_pnl = (long_ce.exit_premium - long_ce.entry_premium) * self.quantity
        short_pe.gross_pnl = (short_pe.entry_premium - short_pe.exit_premium) * self.quantity
        long_pe.gross_pnl = (long_pe.exit_premium - long_pe.entry_premium) * self.quantity

        # Exit costs (4 legs)
        exit_costs = sum(
            calculate_costs(leg.exit_premium, self.quantity, is_entry=False)["total"]
            for leg in all_legs
        )

        total_cost = entry_costs + exit_costs
        for leg in all_legs:
            leg.costs = total_cost / 4

        trade.gross_pnl = sum(leg.gross_pnl for leg in all_legs)
        trade.total_costs = total_cost
        trade.net_pnl = trade.gross_pnl - trade.total_costs
        trade.spot_at_exit = day_data["close"].iloc[-1]

        return trade

    def _close_all_legs(self, trade: IronCondorTrade, ts, reason: str):
        """Close all open legs at current premium."""
        for leg in [trade.short_ce, trade.long_ce, trade.short_pe, trade.long_pe]:
            if leg and leg.is_open:
                leg.exit_premium = leg.current_premium
                leg.exit_time = ts
                leg.status = LegStatus.EXITED_TARGET if "Target" in reason else \
                             LegStatus.EXITED_BREACH if "Breach" in reason else \
                             LegStatus.EXITED_SL if "SL" in reason else \
                             LegStatus.EXITED_MAXLOSS if "Max" in reason else \
                             LegStatus.EXITED_EOD
        trade.exit_time = ts
        trade.exit_reason = reason

    def _compute_results(self) -> BacktestResult:
        """Compute summary statistics."""
        result = BacktestResult(
            trades=self.trades,
            equity_curve=self.equity_curve,
            daily_pnl=self.daily_pnl,
            initial_capital=INITIAL_CAPITAL,
            final_capital=self.capital,
            skipped_days=self.skipped_days,
        )

        if not self.trades:
            return result

        pnls = [t.net_pnl for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        result.total_trades = len(self.trades)
        result.winners = len(wins)
        result.losers = len(losses)
        result.win_rate = len(wins) / len(pnls) * 100 if pnls else 0
        result.total_gross_pnl = sum(t.gross_pnl for t in self.trades)
        result.total_costs = sum(t.total_costs for t in self.trades)
        result.total_net_pnl = sum(pnls)
        result.roi_pct = (result.total_net_pnl / INITIAL_CAPITAL) * 100
        result.avg_win = np.mean(wins) if wins else 0
        result.avg_loss = np.mean(losses) if losses else 0
        result.best_trade = max(pnls) if pnls else 0
        result.worst_trade = min(pnls) if pnls else 0

        total_wins = sum(wins)
        total_losses = abs(sum(losses))
        result.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100
        result.max_drawdown_pct = np.max(drawdown)
        result.max_drawdown_inr = np.max(peak - equity)

        daily_returns = pd.Series(pnls) / INITIAL_CAPITAL
        if daily_returns.std() > 0:
            result.sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(250)

        if self.daily_pnl:
            df_daily = pd.DataFrame(self.daily_pnl)
            df_daily["date"] = pd.to_datetime(df_daily["date"])
            df_daily["month"] = df_daily["date"].dt.to_period("M")
            monthly = df_daily.groupby("month")["net_pnl"].sum()
            monthly_pct = (monthly / INITIAL_CAPITAL) * 100
            result.best_month = monthly_pct.max()
            result.worst_month = monthly_pct.min()
            result.avg_monthly_return = monthly_pct.mean()

        # Consecutive wins/losses
        streak_w = streak_l = max_w = max_l = 0
        for p in pnls:
            if p > 0:
                streak_w += 1; streak_l = 0
                max_w = max(max_w, streak_w)
            else:
                streak_l += 1; streak_w = 0
                max_l = max(max_l, streak_l)
        result.max_consecutive_wins = max_w
        result.max_consecutive_losses = max_l

        # Iron Condor specific stats
        credits = [t.total_net_credit for t in self.trades]
        result.avg_credit_collected = np.mean(credits) if credits else 0

        return result

    def print_summary(self, result: BacktestResult):
        """Print formatted summary."""
        print("\n" + "═" * 70)
        print("  IRON CONDOR — BANK NIFTY — BACKTEST RESULTS")
        print("═" * 70)

        print(f"\n  {'Period:':<30} {len(self.daily_pnl)} trading days")
        print(f"  {'Skipped (filters):':<30} {result.skipped_days} days")
        print(f"  {'Initial Capital:':<30} ₹{result.initial_capital:>12,.0f}")
        print(f"  {'Final Capital:':<30} ₹{result.final_capital:>12,.0f}")

        print(f"\n  {'─' * 45} P&L {'─' * 20}")
        print(f"  {'Total Trades:':<30} {result.total_trades:>12}")
        print(f"  {'Winners:':<30} {result.winners:>12} ({result.win_rate:.1f}%)")
        print(f"  {'Losers:':<30} {result.losers:>12} ({100-result.win_rate:.1f}%)")
        print(f"  {'Gross P&L:':<30} ₹{result.total_gross_pnl:>12,.0f}")
        print(f"  {'Total Costs:':<30} ₹{result.total_costs:>12,.0f}")
        print(f"  {'Net P&L:':<30} ₹{result.total_net_pnl:>12,.0f}")
        print(f"  {'ROI:':<30} {result.roi_pct:>12.1f}%")

        print(f"\n  {'─' * 45} Risk {'─' * 19}")
        print(f"  {'Max Drawdown:':<30} {result.max_drawdown_pct:>11.2f}%  (₹{result.max_drawdown_inr:,.0f})")
        print(f"  {'Profit Factor:':<30} {result.profit_factor:>12.2f}")
        print(f"  {'Sharpe Ratio:':<30} {result.sharpe_ratio:>12.2f}")

        print(f"\n  {'─' * 45} Trades {'─' * 18}")
        print(f"  {'Avg Win:':<30} ₹{result.avg_win:>12,.0f}")
        print(f"  {'Avg Loss:':<30} ₹{result.avg_loss:>12,.0f}")
        print(f"  {'Best Trade:':<30} ₹{result.best_trade:>12,.0f}")
        print(f"  {'Worst Trade:':<30} ₹{result.worst_trade:>12,.0f}")
        print(f"  {'Max Consec Wins:':<30} {result.max_consecutive_wins:>12}")
        print(f"  {'Max Consec Losses:':<30} {result.max_consecutive_losses:>12}")
        print(f"  {'Avg Credit Collected:':<30} {result.avg_credit_collected:>11.1f} pts")

        print(f"\n  {'─' * 45} Monthly {'─' * 17}")
        print(f"  {'Best Month:':<30} {result.best_month:>11.2f}%")
        print(f"  {'Worst Month:':<30} {result.worst_month:>11.2f}%")
        print(f"  {'Avg Monthly Return:':<30} {result.avg_monthly_return:>11.2f}%")

        # Exit reason breakdown
        if self.trades:
            reasons = {}
            for t in self.trades:
                reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
            print(f"\n  {'─' * 45} Exits {'─' * 19}")
            for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
                pct = count / len(self.trades) * 100
                print(f"  {reason:<30} {count:>8} ({pct:.1f}%)")

        print("\n" + "═" * 70)
