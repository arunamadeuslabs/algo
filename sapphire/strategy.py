"""
Sapphire Strategy Engine
=========================
Paired theta-focused short strangle with dynamic trailing stop loss.
Sells OTM Call + Put at market open, trails both legs, exits at EOD.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple
from datetime import datetime, time

from config import (
    INITIAL_CAPITAL, OPTION_LOT_SIZE, NUM_LOTS,
    ENTRY_TIME, SQUARE_OFF_TIME, TRADING_START, TRADING_END,
    OTM_OFFSET_CE, OTM_OFFSET_PE, STRIKE_ROUNDING, DAYS_TO_EXPIRY,
    INITIAL_SL_PCT, TRAIL_ACTIVATE_PCT, TRAIL_STEP_PCT, TRAIL_LOCK_PCT,
    DEEP_PROFIT_PCT, DEEP_TRAIL_LOCK_PCT,
    COMBINED_SL_PCT, MAX_LOSS_PER_DAY,
    MOMENTUM_THRESHOLD, MOMENTUM_SHIFT_ENABLED, MOMENTUM_ROLL_OFFSET,
    RE_ENTRY_ENABLED,
    INCLUDE_COSTS, BROKERAGE_PER_ORDER, SLIPPAGE_POINTS,
    STT_RATE, EXCHANGE_CHARGES, GST_RATE, SEBI_CHARGES, STAMP_DUTY,
    NIFTY_IV, RISK_FREE_RATE,
)
from data_utils import (
    black_scholes_price, bs_greeks, get_atm_strike, get_otm_strikes,
    compute_atr,
)


# ─── Enums & Data Classes ───────────────────────────────────────────────────

class LegStatus(Enum):
    OPEN = "OPEN"
    TRAILED = "TRAILED"
    EXITED_SL = "EXITED_SL"
    EXITED_TARGET = "EXITED_TARGET"
    EXITED_EOD = "EXITED_EOD"
    EXITED_COMBINED = "EXITED_COMBINED"
    EXITED_MAXLOSS = "EXITED_MAXLOSS"
    ROLLED = "ROLLED"


@dataclass
class Leg:
    """Single leg of the strangle (CE or PE)."""
    option_type: str               # "CE" or "PE"
    strike: float
    entry_premium: float
    current_premium: float = 0.0
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    exit_premium: float = 0.0
    status: LegStatus = LegStatus.OPEN
    
    # Trailing SL tracking
    sl_premium: float = 0.0        # Current SL level (in premium)
    min_premium_seen: float = 999  # Lowest premium (max profit)
    trail_active: bool = False
    deep_trail_active: bool = False
    
    # P&L
    gross_pnl: float = 0.0
    costs: float = 0.0
    net_pnl: float = 0.0
    
    @property
    def is_open(self) -> bool:
        return self.status == LegStatus.OPEN


@dataclass
class StrangleTrade:
    """One day's strangle trade (CE + PE legs)."""
    date: datetime
    entry_spot: float
    ce_leg: Optional[Leg] = None
    pe_leg: Optional[Leg] = None
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    
    # Combined tracking
    entry_combined_premium: float = 0.0
    max_premium_collected: float = 0.0
    
    # P&L
    gross_pnl: float = 0.0
    total_costs: float = 0.0
    net_pnl: float = 0.0
    
    # Exit reason
    exit_reason: str = ""
    
    # Momentum
    momentum_detected: bool = False
    spot_at_exit: float = 0.0
    
    @property
    def is_winner(self) -> bool:
        return self.net_pnl > 0

    @property
    def max_spot_move(self) -> float:
        return abs(self.spot_at_exit - self.entry_spot) if self.spot_at_exit else 0


@dataclass
class BacktestResult:
    """Aggregated backtest results."""
    trades: List[StrangleTrade]
    equity_curve: List[float]
    daily_pnl: List[dict]
    initial_capital: float
    final_capital: float
    
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


# ─── Transaction Cost Calculator ────────────────────────────────────────────

def calculate_costs(premium: float, quantity: int, is_entry: bool = True) -> dict:
    """Calculate all transaction costs for one leg."""
    if not INCLUDE_COSTS:
        return {"total": 0, "brokerage": 0, "stt": 0, "exchange": 0, "gst": 0, "sebi": 0, "stamp": 0, "slippage": 0}
    
    turnover = premium * quantity
    
    brokerage = min(BROKERAGE_PER_ORDER, turnover * 0.0003)  # 0.03% or ₹20 whichever lower
    stt = turnover * STT_RATE if not is_entry else 0  # STT only on sell side (exit for short)
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


# ─── Core Strangle Backtest Engine ──────────────────────────────────────────

class SapphireBacktest:
    """
    Nifty Sapphire Intraday Short Strangle Backtest Engine.
    
    Logic:
    1. At 9:20 AM, sell OTM CE + OTM PE (strangle)
    2. Compute premiums using Black-Scholes
    3. Apply dynamic trailing SL on each leg
    4. Optionally shift legs on momentum
    5. Exit at 15:25 or on SL hit
    """
    
    def __init__(self):
        self.capital = INITIAL_CAPITAL
        self.lot_size = OPTION_LOT_SIZE
        self.quantity = OPTION_LOT_SIZE * NUM_LOTS
        self.trades: List[StrangleTrade] = []
        self.equity_curve: List[float] = [INITIAL_CAPITAL]
        self.daily_pnl: List[dict] = []
        
    def run(self, df: pd.DataFrame) -> BacktestResult:
        """
        Run backtest on 5-min OHLCV data.
        df must have: open, high, low, close, volume columns with datetime index.
        """
        print("\n" + "=" * 70)
        print("  SAPPHIRE INTRADAY SHORT STRANGLE BACKTEST")
        print("=" * 70)
        print(f"  Capital: ₹{INITIAL_CAPITAL:,.0f}")
        print(f"  CE Offset: {OTM_OFFSET_CE} pts | PE Offset: {OTM_OFFSET_PE} pts")
        print(f"  Strategy: Sell OTM CE + PE at {ENTRY_TIME}, Exit by {SQUARE_OFF_TIME}")
        print(f"  Trailing SL: Init {INITIAL_SL_PCT*100:.0f}% → Trail {TRAIL_LOCK_PCT*100:.0f}% → Deep {DEEP_TRAIL_LOCK_PCT*100:.0f}%")
        print("=" * 70)
        
        # Group data by trading day
        df = df.copy()
        df["date"] = df.index.date
        trading_days = df.groupby("date")
        
        day_count = 0
        
        for date, day_data in trading_days:
            day_count += 1
            if len(day_data) < 10:
                continue  # Skip days with insufficient data
            
            trade = self._process_day(date, day_data)
            if trade is not None:
                self.trades.append(trade)
                self.capital += trade.net_pnl
                self.equity_curve.append(self.capital)
                
                self.daily_pnl.append({
                    "date": date,
                    "gross_pnl": trade.gross_pnl,
                    "costs": trade.total_costs,
                    "net_pnl": trade.net_pnl,
                    "capital": self.capital,
                    "exit_reason": trade.exit_reason,
                    "ce_strike": trade.ce_leg.strike if trade.ce_leg else 0,
                    "pe_strike": trade.pe_leg.strike if trade.pe_leg else 0,
                    "entry_spot": trade.entry_spot,
                    "exit_spot": trade.spot_at_exit,
                    "ce_entry_prem": trade.ce_leg.entry_premium if trade.ce_leg else 0,
                    "pe_entry_prem": trade.pe_leg.entry_premium if trade.pe_leg else 0,
                    "ce_exit_prem": trade.ce_leg.exit_premium if trade.ce_leg else 0,
                    "pe_exit_prem": trade.pe_leg.exit_premium if trade.pe_leg else 0,
                    "momentum": trade.momentum_detected,
                })
        
        if day_count % 50 == 0:
            print(f"  Processed {day_count} days...")
        
        print(f"\n  Total trading days processed: {day_count}")
        print(f"  Total trades taken: {len(self.trades)}")
        
        return self._compute_results()
    
    def _process_day(self, date, day_data: pd.DataFrame) -> Optional[StrangleTrade]:
        """Process one trading day."""
        
        # Find entry bar (9:20)
        entry_hour, entry_min = map(int, ENTRY_TIME.split(":"))
        exit_hour, exit_min = map(int, SQUARE_OFF_TIME.split(":"))
        
        entry_bar = None
        for ts in day_data.index:
            if ts.hour == entry_hour and ts.minute >= entry_min:
                entry_bar = ts
                break
            elif ts.hour > entry_hour:
                entry_bar = ts
                break
        
        if entry_bar is None:
            return None
        
        entry_spot = day_data.loc[entry_bar, "close"]
        iv_today = day_data["daily_iv"].iloc[0] / 100.0 if "daily_iv" in day_data.columns else NIFTY_IV / 100.0
        r = RISK_FREE_RATE / 100.0
        
        # Determine strikes
        ce_strike, pe_strike = get_otm_strikes(
            entry_spot, OTM_OFFSET_CE, OTM_OFFSET_PE, STRIKE_ROUNDING
        )
        
        # Compute entry premiums
        dte = DAYS_TO_EXPIRY / 365.0
        ce_entry_prem = black_scholes_price(entry_spot, ce_strike, dte, iv_today, r, "CE")
        pe_entry_prem = black_scholes_price(entry_spot, pe_strike, dte, iv_today, r, "PE")
        
        # Add slippage to entry (premium goes against us)
        ce_entry_prem = max(ce_entry_prem, 2.0)
        pe_entry_prem = max(pe_entry_prem, 2.0)
        
        # Create legs
        ce_leg = Leg(
            option_type="CE",
            strike=ce_strike,
            entry_premium=ce_entry_prem,
            current_premium=ce_entry_prem,
            entry_time=entry_bar,
            sl_premium=ce_entry_prem * (1 + INITIAL_SL_PCT),  # SL at 40% rise
            min_premium_seen=ce_entry_prem,
        )
        
        pe_leg = Leg(
            option_type="PE",
            strike=pe_strike,
            entry_premium=pe_entry_prem,
            current_premium=pe_entry_prem,
            entry_time=entry_bar,
            sl_premium=pe_entry_prem * (1 + INITIAL_SL_PCT),
            min_premium_seen=pe_entry_prem,
        )
        
        trade = StrangleTrade(
            date=date,
            entry_spot=entry_spot,
            ce_leg=ce_leg,
            pe_leg=pe_leg,
            entry_time=entry_bar,
            entry_combined_premium=ce_entry_prem + pe_entry_prem,
            max_premium_collected=ce_entry_prem + pe_entry_prem,
        )
        
        # Entry costs (2 legs × sell)
        entry_cost_ce = calculate_costs(ce_entry_prem, self.quantity, is_entry=True)
        entry_cost_pe = calculate_costs(pe_entry_prem, self.quantity, is_entry=True)
        
        # Process each bar after entry
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
            
            # IV adjustment (spike on large spot moves — this is critical for realism)
            spot_move_from_entry = abs(spot - trade.entry_spot)
            spot_move_pct = spot_move_from_entry / trade.entry_spot * 100
            
            iv_adj = iv_today
            # Intraday IV spike when Nifty moves significantly
            if spot_move_pct > 1.0:
                iv_adj *= (1 + spot_move_pct * 0.25)  # Heavy IV spike on 1%+ moves
            elif spot_move_pct > 0.5:
                iv_adj *= (1 + spot_move_pct * 0.15)  # Moderate IV spike
            elif spot_move_pct > 0.3:
                iv_adj *= (1 + spot_move_pct * 0.08)  # Mild IV lift
            
            # Add random IV noise (market microstructure)
            iv_adj *= (1 + np.random.normal(0, 0.03))
            
            # Update CE premium using high (worst case for shorts)
            if ce_leg.is_open:
                ce_prem_current = black_scholes_price(spot, ce_strike, remaining_dte, iv_adj, r, "CE")
                ce_prem_worst = black_scholes_price(high, ce_strike, remaining_dte, iv_adj, r, "CE")
                ce_leg.current_premium = ce_prem_current
                ce_leg.min_premium_seen = min(ce_leg.min_premium_seen, ce_prem_current)
            
            # Update PE premium using low (worst case for shorts)
            if pe_leg.is_open:
                pe_prem_current = black_scholes_price(spot, pe_strike, remaining_dte, iv_adj, r, "PE")
                pe_prem_worst = black_scholes_price(low, pe_strike, remaining_dte, iv_adj, r, "PE")
                pe_leg.current_premium = pe_prem_current
                pe_leg.min_premium_seen = min(pe_leg.min_premium_seen, pe_prem_current)
            
            # --- Check Trailing SL on CE Leg ---
            if ce_leg.is_open:
                self._update_trailing_sl(ce_leg, ce_prem_worst)
                if ce_prem_worst >= ce_leg.sl_premium:
                    ce_leg.exit_premium = ce_leg.sl_premium
                    ce_leg.exit_time = ts
                    ce_leg.status = LegStatus.EXITED_SL
            
            # --- Check Trailing SL on PE Leg ---
            if pe_leg.is_open:
                self._update_trailing_sl(pe_leg, pe_prem_worst)
                if pe_prem_worst >= pe_leg.sl_premium:
                    pe_leg.exit_premium = pe_leg.sl_premium
                    pe_leg.exit_time = ts
                    pe_leg.status = LegStatus.EXITED_SL
            
            # --- Check Combined SL ---
            if ce_leg.is_open or pe_leg.is_open:
                current_combined = (
                    (ce_leg.current_premium if ce_leg.is_open else ce_leg.exit_premium) +
                    (pe_leg.current_premium if pe_leg.is_open else pe_leg.exit_premium)
                )
                if current_combined > trade.entry_combined_premium * (1 + COMBINED_SL_PCT):
                    if ce_leg.is_open:
                        ce_leg.exit_premium = ce_leg.current_premium
                        ce_leg.exit_time = ts
                        ce_leg.status = LegStatus.EXITED_COMBINED
                    if pe_leg.is_open:
                        pe_leg.exit_premium = pe_leg.current_premium
                        pe_leg.exit_time = ts
                        pe_leg.status = LegStatus.EXITED_COMBINED
                    trade.exit_reason = "Combined SL"
            
            # --- Check Momentum Shift ---
            if MOMENTUM_SHIFT_ENABLED and (ce_leg.is_open and pe_leg.is_open):
                spot_move = spot - trade.entry_spot
                if abs(spot_move) >= MOMENTUM_THRESHOLD:
                    trade.momentum_detected = True
            
            # --- Max Daily Loss Check ---
            running_pnl = self._calc_running_pnl(trade)
            if running_pnl < -MAX_LOSS_PER_DAY:
                if ce_leg.is_open:
                    ce_leg.exit_premium = ce_leg.current_premium
                    ce_leg.exit_time = ts
                    ce_leg.status = LegStatus.EXITED_MAXLOSS
                if pe_leg.is_open:
                    pe_leg.exit_premium = pe_leg.current_premium
                    pe_leg.exit_time = ts
                    pe_leg.status = LegStatus.EXITED_MAXLOSS
                trade.exit_reason = "Max Daily Loss"
            
            # --- EOD Exit ---
            if ts.hour == exit_hour and ts.minute >= exit_min:
                if ce_leg.is_open:
                    ce_leg.exit_premium = ce_leg.current_premium
                    ce_leg.exit_time = ts
                    ce_leg.status = LegStatus.EXITED_EOD
                if pe_leg.is_open:
                    pe_leg.exit_premium = pe_leg.current_premium
                    pe_leg.exit_time = ts
                    pe_leg.status = LegStatus.EXITED_EOD
                if not trade.exit_reason:
                    trade.exit_reason = "EOD Square-off"
            elif ts.hour > exit_hour:
                if ce_leg.is_open:
                    ce_leg.exit_premium = ce_leg.current_premium
                    ce_leg.exit_time = ts
                    ce_leg.status = LegStatus.EXITED_EOD
                if pe_leg.is_open:
                    pe_leg.exit_premium = pe_leg.current_premium
                    pe_leg.exit_time = ts
                    pe_leg.status = LegStatus.EXITED_EOD
                if not trade.exit_reason:
                    trade.exit_reason = "EOD Square-off"
            
            # If both legs closed, finalize
            if not ce_leg.is_open and not pe_leg.is_open:
                break
        
        # Force close any remaining open legs
        if ce_leg.is_open:
            ce_leg.exit_premium = ce_leg.current_premium
            ce_leg.exit_time = day_data.index[-1]
            ce_leg.status = LegStatus.EXITED_EOD
        if pe_leg.is_open:
            pe_leg.exit_premium = pe_leg.current_premium
            pe_leg.exit_time = day_data.index[-1]
            pe_leg.status = LegStatus.EXITED_EOD
        if not trade.exit_reason:
            trade.exit_reason = "EOD Square-off"
        
        # Calculate P&L
        # Short strangle P&L = (entry_premium - exit_premium) × quantity
        ce_leg.gross_pnl = (ce_leg.entry_premium - ce_leg.exit_premium) * self.quantity
        pe_leg.gross_pnl = (pe_leg.entry_premium - pe_leg.exit_premium) * self.quantity
        
        # Calculate exit costs
        exit_cost_ce = calculate_costs(ce_leg.exit_premium, self.quantity, is_entry=False)
        exit_cost_pe = calculate_costs(pe_leg.exit_premium, self.quantity, is_entry=False)
        
        ce_leg.costs = entry_cost_ce["total"] + exit_cost_ce["total"]
        pe_leg.costs = entry_cost_pe["total"] + exit_cost_pe["total"]
        ce_leg.net_pnl = ce_leg.gross_pnl - ce_leg.costs
        pe_leg.net_pnl = pe_leg.gross_pnl - pe_leg.costs
        
        trade.gross_pnl = ce_leg.gross_pnl + pe_leg.gross_pnl
        trade.total_costs = ce_leg.costs + pe_leg.costs
        trade.net_pnl = trade.gross_pnl - trade.total_costs
        trade.spot_at_exit = day_data["close"].iloc[-1]
        
        if not trade.exit_reason:
            if ce_leg.status == LegStatus.EXITED_SL or pe_leg.status == LegStatus.EXITED_SL:
                trade.exit_reason = "Trail SL Hit"
        
        return trade
    
    def _update_trailing_sl(self, leg: Leg, worst_premium: float):
        """
        Dynamic trailing SL logic:
        Phase 1: Initial SL (premium rises 40% = stop)
        Phase 2: After 20% decay, trail to lock 50% of gains
        Phase 3: After 50% decay, tight trail to lock 70%
        """
        entry_prem = leg.entry_premium
        current_prem = leg.current_premium
        
        # How much premium has decayed (positive = good for us)
        decay_pct = (entry_prem - leg.min_premium_seen) / entry_prem if entry_prem > 0 else 0
        
        # Phase 3: Deep profit trailing
        if decay_pct >= DEEP_PROFIT_PCT and not leg.deep_trail_active:
            leg.deep_trail_active = True
            leg.trail_active = True
        
        if leg.deep_trail_active:
            # Lock 70% of max profit seen
            max_gain = entry_prem - leg.min_premium_seen
            new_sl = entry_prem - (max_gain * DEEP_TRAIL_LOCK_PCT)
            leg.sl_premium = min(leg.sl_premium, max(new_sl, leg.min_premium_seen * 1.15))
            return
        
        # Phase 2: Standard trailing
        if decay_pct >= TRAIL_ACTIVATE_PCT and not leg.trail_active:
            leg.trail_active = True
        
        if leg.trail_active:
            # Lock 50% of max profit seen
            max_gain = entry_prem - leg.min_premium_seen
            new_sl = entry_prem - (max_gain * TRAIL_LOCK_PCT)
            leg.sl_premium = min(leg.sl_premium, max(new_sl, leg.min_premium_seen * 1.25))
            return
        
        # Phase 1: Initial SL stays at entry * (1 + 40%)
        # Already set at entry
    
    def _calc_running_pnl(self, trade: StrangleTrade) -> float:
        """Calculate current unrealized + realized P&L."""
        pnl = 0
        if trade.ce_leg:
            if trade.ce_leg.is_open:
                pnl += (trade.ce_leg.entry_premium - trade.ce_leg.current_premium) * self.quantity
            else:
                pnl += trade.ce_leg.gross_pnl
        if trade.pe_leg:
            if trade.pe_leg.is_open:
                pnl += (trade.pe_leg.entry_premium - trade.pe_leg.current_premium) * self.quantity
            else:
                pnl += trade.pe_leg.gross_pnl
        return pnl
    
    def _compute_results(self) -> BacktestResult:
        """Compute summary statistics."""
        result = BacktestResult(
            trades=self.trades,
            equity_curve=self.equity_curve,
            daily_pnl=self.daily_pnl,
            initial_capital=INITIAL_CAPITAL,
            final_capital=self.capital,
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
        
        # Profit factor
        total_wins = sum(wins)
        total_losses = abs(sum(losses))
        result.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Max drawdown
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100
        result.max_drawdown_pct = np.max(drawdown)
        result.max_drawdown_inr = np.max(peak - equity)
        
        # Sharpe ratio (annualized, assuming ~250 trading days)
        daily_returns = pd.Series(pnls) / INITIAL_CAPITAL
        if daily_returns.std() > 0:
            result.sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(250)
        
        # Monthly returns
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
                streak_w += 1
                streak_l = 0
                max_w = max(max_w, streak_w)
            else:
                streak_l += 1
                streak_w = 0
                max_l = max(max_l, streak_l)
        result.max_consecutive_wins = max_w
        result.max_consecutive_losses = max_l
        
        return result
    
    def print_summary(self, result: BacktestResult):
        """Print formatted summary."""
        print("\n" + "═" * 70)
        print("  💎 SAPPHIRE STRATEGY — BACKTEST RESULTS")
        print("═" * 70)
        
        print(f"\n  {'Period:':<28} {len(self.daily_pnl)} trading days")
        print(f"  {'Initial Capital:':<28} ₹{result.initial_capital:>12,.0f}")
        print(f"  {'Final Capital:':<28} ₹{result.final_capital:>12,.0f}")
        
        print(f"\n  {'─' * 45} P&L {'─' * 20}")
        print(f"  {'Total Trades:':<28} {result.total_trades:>12}")
        print(f"  {'Winners:':<28} {result.winners:>12} ({result.win_rate:.1f}%)")
        print(f"  {'Losers:':<28} {result.losers:>12} ({100-result.win_rate:.1f}%)")
        print(f"  {'Gross P&L:':<28} ₹{result.total_gross_pnl:>12,.0f}")
        print(f"  {'Total Costs:':<28} ₹{result.total_costs:>12,.0f}")
        print(f"  {'Net P&L:':<28} ₹{result.total_net_pnl:>12,.0f}")
        print(f"  {'ROI:':<28} {result.roi_pct:>12.1f}%")
        
        print(f"\n  {'─' * 45} Risk {'─' * 19}")
        print(f"  {'Max Drawdown:':<28} {result.max_drawdown_pct:>11.2f}%  (₹{result.max_drawdown_inr:,.0f})")
        print(f"  {'Profit Factor:':<28} {result.profit_factor:>12.2f}")
        print(f"  {'Sharpe Ratio:':<28} {result.sharpe_ratio:>12.2f}")
        
        print(f"\n  {'─' * 45} Trades {'─' * 18}")
        print(f"  {'Avg Win:':<28} ₹{result.avg_win:>12,.0f}")
        print(f"  {'Avg Loss:':<28} ₹{result.avg_loss:>12,.0f}")
        print(f"  {'Best Trade:':<28} ₹{result.best_trade:>12,.0f}")
        print(f"  {'Worst Trade:':<28} ₹{result.worst_trade:>12,.0f}")
        print(f"  {'Max Consec Wins:':<28} {result.max_consecutive_wins:>12}")
        print(f"  {'Max Consec Losses:':<28} {result.max_consecutive_losses:>12}")
        
        print(f"\n  {'─' * 45} Monthly {'─' * 17}")
        print(f"  {'Best Month:':<28} {result.best_month:>11.2f}%")
        print(f"  {'Worst Month:':<28} {result.worst_month:>11.2f}%")
        print(f"  {'Avg Monthly Return:':<28} {result.avg_monthly_return:>11.2f}%")
        
        # Exit reason breakdown
        if self.trades:
            reasons = {}
            for t in self.trades:
                reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
            print(f"\n  {'─' * 45} Exits {'─' * 19}")
            for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
                pct = count / len(self.trades) * 100
                print(f"  {reason:<28} {count:>8} ({pct:.1f}%)")
        
        print("\n" + "═" * 70)
