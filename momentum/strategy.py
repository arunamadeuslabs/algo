"""
Dual-Confirmation Momentum Strategy Engine
============================================
Intraday momentum strategy for Nifty futures using:
  - Dual signal confirmation: RSI/MACD crossover + price action
  - Dynamic trailing stops based on ATR
  - Partial profit booking: 50% at 1:1 R:R, trail remaining

Entry Logic:
  LONG: MACD bullish crossover + RSI > 35 (not oversold) + price above EMA
        + strong bullish candle + volume surge + ADX trending
  SHORT: MACD bearish crossover + RSI < 65 (not overbought) + price below EMA
         + strong bearish candle + volume surge + ADX trending

Exit Logic:
  1. Partial exit: 50% at 1:1 risk-reward
  2. Trail remaining 50% with dynamic ATR-based trailing stop
  3. Full exit at 15:20 EOD or max daily loss
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List
from datetime import datetime, time

from config import (
    INITIAL_CAPITAL, NIFTY_LOT_SIZE, MAX_LOTS,
    TRADING_START, TRADING_END, SQUARE_OFF_TIME,
    RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD, RSI_EXTREME_OB, RSI_EXTREME_OS,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    EMA_FAST, EMA_SLOW,
    ATR_PERIOD, ATR_SL_MULTIPLIER, MIN_SL_POINTS, MAX_SL_POINTS,
    PARTIAL_EXIT_PCT, PARTIAL_TARGET_RR, TRAIL_REMAINING,
    TRAIL_ACTIVATE_POINTS, TRAIL_STEP_POINTS, TRAIL_LOCK_PCT,
    TIGHT_TRAIL_ACTIVATE, TIGHT_TRAIL_LOCK_PCT,
    VOLUME_SURGE_MULTIPLIER, ADX_MIN_TREND, MOMENTUM_THRESHOLD,
    ENTRY_COOLDOWN_BARS, MAX_TRADES_PER_DAY, MAX_LOSS_PER_DAY,
    INCLUDE_COSTS, BROKERAGE_PER_ORDER, SLIPPAGE_POINTS,
    STT_RATE, EXCHANGE_CHARGES, GST_RATE, SEBI_CHARGES, STAMP_DUTY,
    STRUCTURE_BREAK_BUFFER,
)
from data_utils import compute_indicators


# ─── Enums ───────────────────────────────────────────────────────────────────

class Signal(Enum):
    NONE = 0
    LONG = 1
    SHORT = -1


class TradeStatus(Enum):
    OPEN = "OPEN"
    PARTIAL_EXIT = "PARTIAL_EXIT"
    TARGET_HIT = "TARGET_HIT"
    SL_HIT = "SL_HIT"
    TRAILING_SL = "TRAILING_SL"
    TIME_EXIT = "TIME_EXIT"
    MAX_LOSS_EXIT = "MAX_LOSS_EXIT"


# ─── Data Classes ────────────────────────────────────────────────────────────

@dataclass
class Trade:
    """Single momentum trade."""
    entry_time: Optional[datetime] = None
    direction: Signal = Signal.NONE
    entry_price: float = 0.0
    entry_lots: int = 0
    current_lots: int = 0           # Remaining after partial exit
    sl_price: float = 0.0
    initial_sl_points: float = 0.0  # Used for R:R calculation
    target_price: float = 0.0       # Partial exit target (1:1 R:R)

    exit_time: Optional[datetime] = None
    exit_price: float = 0.0
    status: TradeStatus = TradeStatus.OPEN

    # Partial exit tracking
    partial_exited: bool = False
    partial_exit_price: float = 0.0
    partial_exit_time: Optional[datetime] = None
    partial_exit_lots: int = 0
    partial_pnl: float = 0.0

    # Trailing stop
    trailing_sl: float = 0.0
    max_favorable: float = 0.0
    trail_active: bool = False
    tight_trail_active: bool = False

    # P&L
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
        return self.status in (TradeStatus.OPEN, TradeStatus.PARTIAL_EXIT)

    @property
    def quantity(self) -> int:
        return self.current_lots * NIFTY_LOT_SIZE

    @property
    def full_quantity(self) -> int:
        return self.entry_lots * NIFTY_LOT_SIZE


@dataclass
class BacktestResult:
    """Aggregated backtest results."""
    trades: List[Trade]
    equity_curve: List[float]
    daily_pnl: List[dict]
    initial_capital: float
    final_capital: float

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
    partial_exit_count: int = 0
    full_trail_wins: int = 0


# ─── Transaction Cost Calculator ────────────────────────────────────────────

def calculate_costs(price: float, quantity: int, is_entry: bool = True) -> dict:
    """Calculate transaction costs for futures trades."""
    if not INCLUDE_COSTS:
        return {"total": 0, "brokerage": 0, "stt": 0, "exchange": 0,
                "gst": 0, "sebi": 0, "stamp": 0, "slippage": 0}

    turnover = price * quantity

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


# ─── Core Momentum Backtest Engine ──────────────────────────────────────────

class MomentumBacktest:
    """
    Dual-Confirmation Momentum Strategy Backtest Engine.

    Logic:
    1. Wait for dual confirmation (MACD crossover + RSI + price action)
    2. Enter long/short with ATR-based stop loss
    3. Exit 50% at 1:1 R:R (partial profit booking)
    4. Trail remaining 50% with dynamic trailing stop
    5. Square off at 15:20 or on max daily loss
    """

    def __init__(self):
        self.capital = INITIAL_CAPITAL
        self.lot_size = NIFTY_LOT_SIZE
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [INITIAL_CAPITAL]
        self.daily_pnl: List[dict] = []

    def run(self, df: pd.DataFrame) -> BacktestResult:
        """Run backtest on OHLCV data."""
        print("\n" + "=" * 70)
        print("  DUAL-CONFIRMATION MOMENTUM STRATEGY BACKTEST")
        print("=" * 70)
        print(f"  Capital: ₹{INITIAL_CAPITAL:,.0f}")
        print(f"  Max Lots: {MAX_LOTS}")
        print(f"  Strategy: MACD + RSI dual confirmation + partial exits")
        print(f"  Trailing: ATR-based dynamic trailing stop")
        print("=" * 70)

        # Compute indicators
        df = compute_indicators(df)

        # Group by trading day
        df["date"] = df.index.date
        trading_days = df.groupby("date")

        day_count = 0

        for date_val, day_data in trading_days:
            day_count += 1
            if len(day_data) < 15:
                continue

            day_trades = self._process_day(date_val, day_data)
            for trade in day_trades:
                self.trades.append(trade)
                self.capital += trade.net_pnl
                self.equity_curve.append(self.capital)

                self.daily_pnl.append({
                    "date": date_val,
                    "direction": trade.direction.name,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "gross_pnl": trade.gross_pnl,
                    "costs": trade.costs,
                    "net_pnl": trade.net_pnl,
                    "capital": self.capital,
                    "status": trade.status.value,
                    "partial_exited": trade.partial_exited,
                    "entry_rsi": trade.entry_rsi,
                    "entry_atr": trade.entry_atr,
                })

            if day_count % 50 == 0:
                print(f"  Processed {day_count} days...")

        print(f"\n  Total trading days processed: {day_count}")
        print(f"  Total trades taken: {len(self.trades)}")

        return self._compute_results()

    def _process_day(self, date_val, day_data: pd.DataFrame) -> List[Trade]:
        """Process one trading day. Returns list of closed trades."""
        trades_today = []
        current_trade: Optional[Trade] = None
        daily_loss = 0.0
        last_exit_bar_idx = -ENTRY_COOLDOWN_BARS
        trade_count = 0

        for bar_idx, (ts, row) in enumerate(day_data.iterrows()):
            time_str = f"{ts.hour:02d}:{ts.minute:02d}"

            # --- Manage existing trade ---
            if current_trade is not None and current_trade.is_open:
                current_trade = self._manage_trade(current_trade, row, ts, time_str)

                if not current_trade.is_open:
                    # Trade closed
                    daily_loss += min(0, current_trade.net_pnl)
                    trades_today.append(current_trade)
                    last_exit_bar_idx = bar_idx
                    current_trade = None
                    continue

            # --- Check for new entry ---
            if current_trade is None:
                # Filters
                if time_str < TRADING_START or time_str > TRADING_END:
                    continue
                if trade_count >= MAX_TRADES_PER_DAY:
                    continue
                if bar_idx - last_exit_bar_idx < ENTRY_COOLDOWN_BARS:
                    continue
                if daily_loss <= -MAX_LOSS_PER_DAY:
                    continue

                signal = self._generate_signal(row)
                if signal != Signal.NONE:
                    current_trade = self._enter_trade(row, ts, signal)
                    if current_trade:
                        trade_count += 1

        # Force close any open trade at EOD
        if current_trade is not None and current_trade.is_open:
            self._close_trade(current_trade, day_data.iloc[-1], day_data.index[-1],
                              TradeStatus.TIME_EXIT)
            trades_today.append(current_trade)

        return trades_today

    def _generate_signal(self, row: pd.Series) -> Signal:
        """
        Generate entry signal using dual confirmation:
        1. MACD crossover (primary momentum)
        2. RSI confirmation (not exhausted)
        3. Price action (close vs EMA, strong candle)
        4. Volume surge
        5. ADX trending
        """
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

        # Skip if no MACD crossover
        if not (macd_bull or macd_bear):
            return Signal.NONE

        # Must be trending
        if not trending:
            return Signal.NONE

        # Volume confirmation
        if not volume_surge:
            return Signal.NONE

        # --- LONG Signal ---
        if macd_bull:
            # RSI: not in extreme overbought, must be above oversold
            if rsi >= RSI_EXTREME_OB or rsi < RSI_OVERSOLD:
                return Signal.NONE
            # Price above slow EMA (trend alignment)
            if not trend_up:
                return Signal.NONE
            # Strong bullish candle
            if not bullish_candle:
                return Signal.NONE
            # MACD histogram positive and increasing
            if macd_hist <= 0:
                return Signal.NONE
            return Signal.LONG

        # --- SHORT Signal ---
        if macd_bear:
            if rsi <= RSI_EXTREME_OS or rsi > RSI_OVERBOUGHT:
                return Signal.NONE
            if not trend_down:
                return Signal.NONE
            if not bearish_candle:
                return Signal.NONE
            if macd_hist >= 0:
                return Signal.NONE
            return Signal.SHORT

        return Signal.NONE

    def _enter_trade(self, row: pd.Series, ts, signal: Signal) -> Optional[Trade]:
        """Enter a trade with ATR-based stop loss."""
        price = row["close"]
        atr = row.get("atr", 30)

        if pd.isna(atr) or atr <= 0:
            atr = 30

        # Dynamic SL based on ATR
        sl_points = atr * ATR_SL_MULTIPLIER
        sl_points = max(MIN_SL_POINTS, min(sl_points, MAX_SL_POINTS))

        # Use recent swing for SL placement
        swing_low = row.get("swing_low", price - sl_points)
        swing_high = row.get("swing_high", price + sl_points)

        if signal == Signal.LONG:
            # SL below recent swing low or ATR-based, whichever is closer
            atr_sl = price - sl_points
            structure_sl = swing_low - STRUCTURE_BREAK_BUFFER
            sl_price = max(atr_sl, structure_sl)  # Use tighter SL
            sl_distance = price - sl_price
            target_price = price + sl_distance * PARTIAL_TARGET_RR
        else:
            atr_sl = price + sl_points
            structure_sl = swing_high + STRUCTURE_BREAK_BUFFER
            sl_price = min(atr_sl, structure_sl)
            sl_distance = sl_price - price
            target_price = price - sl_distance * PARTIAL_TARGET_RR

        # Position sizing: risk-based
        risk_per_lot = sl_distance * NIFTY_LOT_SIZE
        max_risk = self.capital * 0.02  # Risk 2% per trade
        lots = min(MAX_LOTS, max(1, int(max_risk / risk_per_lot))) if risk_per_lot > 0 else 1

        trade = Trade(
            entry_time=ts,
            direction=signal,
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

        return trade

    def _manage_trade(self, trade: Trade, row: pd.Series, ts, time_str: str) -> Trade:
        """Manage open trade: check SL, partial exit, trailing, EOD."""
        price = row["close"]
        high = row["high"]
        low = row["low"]

        # --- EOD Square-off ---
        if time_str >= SQUARE_OFF_TIME:
            self._close_trade(trade, row, ts, TradeStatus.TIME_EXIT)
            return trade

        if trade.direction == Signal.LONG:
            # --- Check Stop Loss ---
            if low <= trade.sl_price:
                trade.exit_price = trade.sl_price
                trade.exit_time = ts
                trade.status = TradeStatus.SL_HIT if not trade.partial_exited else TradeStatus.TRAILING_SL
                self._calculate_pnl(trade)
                return trade

            # Track max favorable
            trade.max_favorable = max(trade.max_favorable, high - trade.entry_price)

            # --- Partial Exit at 1:1 R:R ---
            if not trade.partial_exited and high >= trade.target_price:
                self._partial_exit(trade, trade.target_price, ts)

            # --- Trailing Stop (after partial exit) ---
            if trade.partial_exited and TRAIL_REMAINING:
                self._update_trailing_sl_long(trade, price, high)

        else:  # SHORT
            if high >= trade.sl_price:
                trade.exit_price = trade.sl_price
                trade.exit_time = ts
                trade.status = TradeStatus.SL_HIT if not trade.partial_exited else TradeStatus.TRAILING_SL
                self._calculate_pnl(trade)
                return trade

            trade.max_favorable = max(trade.max_favorable, trade.entry_price - low)

            if not trade.partial_exited and low <= trade.target_price:
                self._partial_exit(trade, trade.target_price, ts)

            if trade.partial_exited and TRAIL_REMAINING:
                self._update_trailing_sl_short(trade, price, low)

        return trade

    def _partial_exit(self, trade: Trade, exit_price: float, ts):
        """Exit 50% of position at 1:1 R:R target."""
        partial_lots = max(1, int(trade.entry_lots * PARTIAL_EXIT_PCT))
        partial_qty = partial_lots * NIFTY_LOT_SIZE

        if trade.direction == Signal.LONG:
            trade.partial_pnl = (exit_price - trade.entry_price) * partial_qty
        else:
            trade.partial_pnl = (trade.entry_price - exit_price) * partial_qty

        # Deduct costs for partial exit
        exit_costs = calculate_costs(exit_price, partial_qty, is_entry=False)
        trade.partial_pnl -= exit_costs["total"]

        trade.partial_exited = True
        trade.partial_exit_price = exit_price
        trade.partial_exit_time = ts
        trade.partial_exit_lots = partial_lots
        trade.current_lots = trade.entry_lots - partial_lots
        trade.status = TradeStatus.PARTIAL_EXIT

        # Move SL to breakeven for remaining position
        trade.sl_price = trade.entry_price
        trade.trailing_sl = trade.entry_price

    def _update_trailing_sl_long(self, trade: Trade, price: float, high: float):
        """Update trailing stop for long position."""
        profit_points = high - trade.entry_price

        # Phase 1: Standard trail
        if profit_points >= TRAIL_ACTIVATE_POINTS and not trade.trail_active:
            trade.trail_active = True

        # Phase 2: Tight trail
        if profit_points >= TIGHT_TRAIL_ACTIVATE and not trade.tight_trail_active:
            trade.tight_trail_active = True

        if trade.tight_trail_active:
            new_sl = high - profit_points * (1 - TIGHT_TRAIL_LOCK_PCT)
            trade.sl_price = max(trade.sl_price, new_sl)
            trade.trailing_sl = trade.sl_price
        elif trade.trail_active:
            new_sl = high - profit_points * (1 - TRAIL_LOCK_PCT)
            new_sl = max(new_sl, trade.entry_price + TRAIL_STEP_POINTS)
            trade.sl_price = max(trade.sl_price, new_sl)
            trade.trailing_sl = trade.sl_price

    def _update_trailing_sl_short(self, trade: Trade, price: float, low: float):
        """Update trailing stop for short position."""
        profit_points = trade.entry_price - low

        if profit_points >= TRAIL_ACTIVATE_POINTS and not trade.trail_active:
            trade.trail_active = True

        if profit_points >= TIGHT_TRAIL_ACTIVATE and not trade.tight_trail_active:
            trade.tight_trail_active = True

        if trade.tight_trail_active:
            new_sl = low + profit_points * (1 - TIGHT_TRAIL_LOCK_PCT)
            trade.sl_price = min(trade.sl_price, new_sl)
            trade.trailing_sl = trade.sl_price
        elif trade.trail_active:
            new_sl = low + profit_points * (1 - TRAIL_LOCK_PCT)
            new_sl = min(new_sl, trade.entry_price - TRAIL_STEP_POINTS)
            trade.sl_price = min(trade.sl_price, new_sl)
            trade.trailing_sl = trade.sl_price

    def _close_trade(self, trade: Trade, row: pd.Series, ts, status: TradeStatus):
        """Close trade at current price."""
        trade.exit_price = row["close"]
        trade.exit_time = ts
        trade.status = status
        self._calculate_pnl(trade)

    def _calculate_pnl(self, trade: Trade):
        """Calculate trade P&L including partial exit."""
        remaining_qty = trade.current_lots * NIFTY_LOT_SIZE

        if trade.direction == Signal.LONG:
            remaining_pnl = (trade.exit_price - trade.entry_price) * remaining_qty
        else:
            remaining_pnl = (trade.entry_price - trade.exit_price) * remaining_qty

        # Entry costs (full position)
        entry_costs = calculate_costs(trade.entry_price, trade.full_quantity, is_entry=True)

        # Exit costs for remaining position
        exit_costs = calculate_costs(trade.exit_price, remaining_qty, is_entry=False)

        trade.gross_pnl = trade.partial_pnl + remaining_pnl
        trade.costs = entry_costs["total"] + exit_costs["total"]
        # Note: partial exit costs already deducted from partial_pnl
        trade.net_pnl = trade.gross_pnl - entry_costs["total"] - exit_costs["total"]

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
        result.total_costs = sum(t.costs for t in self.trades)
        result.total_net_pnl = sum(pnls)
        result.roi_pct = (result.total_net_pnl / INITIAL_CAPITAL) * 100
        result.avg_win = np.mean(wins) if wins else 0
        result.avg_loss = np.mean(losses) if losses else 0
        result.best_trade = max(pnls) if pnls else 0
        result.worst_trade = min(pnls) if pnls else 0

        # Partial exit stats
        result.partial_exit_count = sum(1 for t in self.trades if t.partial_exited)
        result.full_trail_wins = sum(1 for t in self.trades
                                      if t.partial_exited and t.net_pnl > t.partial_pnl)

        # Profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        result.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Max drawdown
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100
        result.max_drawdown_pct = np.max(drawdown) if len(drawdown) > 0 else 0
        result.max_drawdown_inr = np.max(peak - equity) if len(peak) > 0 else 0

        # Sharpe ratio
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
        print("  ⚡ DUAL-CONFIRMATION MOMENTUM — BACKTEST RESULTS")
        print("═" * 70)

        print(f"\n  {'Period:':<28} {len(set(d['date'] for d in self.daily_pnl))} trading days")
        print(f"  {'Initial Capital:':<28} ₹{result.initial_capital:>12,.0f}")
        print(f"  {'Final Capital:':<28} ₹{result.final_capital:>12,.0f}")

        print(f"\n  {'─' * 45} P&L {'─' * 20}")
        print(f"  {'Total Trades:':<28} {result.total_trades:>12}")
        print(f"  {'Winners:':<28} {result.winners:>12} ({result.win_rate:.1f}%)")
        print(f"  {'Losers:':<28} {result.losers:>12} ({100 - result.win_rate:.1f}%)")
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

        print(f"\n  {'─' * 45} Partial Exits {'─' * 11}")
        print(f"  {'Partial Exits Taken:':<28} {result.partial_exit_count:>12}")
        print(f"  {'Trail > Partial Profit:':<28} {result.full_trail_wins:>12}")

        print(f"\n  {'─' * 45} Monthly {'─' * 17}")
        print(f"  {'Best Month:':<28} {result.best_month:>11.2f}%")
        print(f"  {'Worst Month:':<28} {result.worst_month:>11.2f}%")
        print(f"  {'Avg Monthly Return:':<28} {result.avg_monthly_return:>11.2f}%")

        # Exit reason breakdown
        if self.trades:
            reasons = {}
            for t in self.trades:
                reasons[t.status.value] = reasons.get(t.status.value, 0) + 1
            print(f"\n  {'─' * 45} Exits {'─' * 19}")
            for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
                pct = count / len(self.trades) * 100
                print(f"  {reason:<28} {count:>8} ({pct:.1f}%)")

        print("\n" + "═" * 70)
