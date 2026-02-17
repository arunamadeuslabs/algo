"""
Supertrend + VWAP Scalping Strategy Engine
===========================================
High-accuracy intraday scalping strategy for Nifty futures using:
  - Supertrend (10,3) for trend direction
  - VWAP for institutional bias
  - 9 EMA for candle confirmation

Entry Logic:
  LONG:  Price > VWAP + Supertrend = Green + Candle above 9 EMA
  SHORT: Price < VWAP + Supertrend = Red  + Candle below 9 EMA

Exit Logic:
  1. Stop Loss: Below Supertrend line (max ₹1,000 per trade)
  2. Target: ₹1,500–₹2,000 per trade
  3. Trailing stop after ₹500+ profit
  4. EOD square off at 15:20

Expected: ~65% win rate, 2-4 trades/day, ~₹5k daily target
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
    SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER,
    EMA_PERIOD,
    MAX_SL_PER_TRADE, TARGET_MIN, TARGET_MAX, RISK_REWARD_RATIO,
    SL_BUFFER_POINTS, MIN_SL_POINTS, MAX_SL_POINTS,
    TRAIL_ACTIVATE_POINTS, TRAIL_LOCK_PCT,
    TIGHT_TRAIL_ACTIVATE, TIGHT_TRAIL_LOCK_PCT,
    ENTRY_COOLDOWN_BARS, MAX_TRADES_PER_DAY, MAX_LOSS_PER_DAY,
    INCLUDE_COSTS, BROKERAGE_PER_ORDER, SLIPPAGE_POINTS,
    STT_RATE, EXCHANGE_CHARGES, GST_RATE, SEBI_CHARGES, STAMP_DUTY,
)
from data_utils import compute_indicators


# ─── Enums ───────────────────────────────────────────────────────────────────

class Signal(Enum):
    NONE = 0
    LONG = 1
    SHORT = -1


class TradeStatus(Enum):
    OPEN = "OPEN"
    TARGET_HIT = "TARGET_HIT"
    SL_HIT = "SL_HIT"
    TRAILING_SL = "TRAILING_SL"
    TIME_EXIT = "TIME_EXIT"
    MAX_LOSS_EXIT = "MAX_LOSS_EXIT"
    DAILY_TARGET_HIT = "DAILY_TARGET_HIT"


# ─── Data Classes ────────────────────────────────────────────────────────────

@dataclass
class Trade:
    """Single scalping trade."""
    entry_time: Optional[datetime] = None
    direction: Signal = Signal.NONE
    entry_price: float = 0.0
    lots: int = 0
    sl_price: float = 0.0
    initial_sl_points: float = 0.0
    target_price: float = 0.0

    exit_time: Optional[datetime] = None
    exit_price: float = 0.0
    status: TradeStatus = TradeStatus.OPEN

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
    entry_supertrend: float = 0.0
    entry_vwap: float = 0.0
    entry_ema: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.status == TradeStatus.OPEN

    @property
    def quantity(self) -> int:
        return self.lots * NIFTY_LOT_SIZE


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
    avg_trades_per_day: float = 0.0
    target_hit_rate: float = 0.0


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


# ─── Core Supertrend Scalping Backtest Engine ───────────────────────────────

class SupertrendBacktest:
    """
    Supertrend + VWAP Scalping Strategy Backtest Engine.

    Logic:
    1. Wait for Supertrend + VWAP + EMA alignment
    2. Enter long/short with SL below Supertrend (max ₹1,000)
    3. Target ₹1,500–₹2,000 based on R:R
    4. Trail remaining with dynamic trailing stop
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
        print("  SUPERTREND + VWAP SCALPING STRATEGY BACKTEST")
        print("=" * 70)
        print(f"  Capital: ₹{INITIAL_CAPITAL:,.0f}")
        print(f"  Max Lots: {MAX_LOTS}")
        print(f"  Strategy: Supertrend ({SUPERTREND_PERIOD},{SUPERTREND_MULTIPLIER}) + VWAP + 9 EMA")
        print(f"  Risk: Max ₹{MAX_SL_PER_TRADE}/trade | Target: ₹{TARGET_MIN}–₹{TARGET_MAX}")
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
                    "entry_supertrend": trade.entry_supertrend,
                    "entry_vwap": trade.entry_vwap,
                    "entry_ema": trade.entry_ema,
                })

            if day_count % 50 == 0:
                print(f"  Processed {day_count} days...")

        print(f"\n  Total trading days processed: {day_count}")
        print(f"  Total trades taken: {len(self.trades)}")

        return self._compute_results()

    def _process_day(self, date_val, day_data: pd.DataFrame) -> List[Trade]:
        """Process one trading day."""
        trades_today = []
        current_trade: Optional[Trade] = None
        daily_loss = 0.0
        daily_profit = 0.0
        last_exit_bar_idx = -ENTRY_COOLDOWN_BARS
        trade_count = 0

        for bar_idx, (ts, row) in enumerate(day_data.iterrows()):
            time_str = f"{ts.hour:02d}:{ts.minute:02d}"

            # --- Manage existing trade ---
            if current_trade is not None and current_trade.is_open:
                current_trade = self._manage_trade(current_trade, row, ts, time_str)

                if not current_trade.is_open:
                    daily_loss += min(0, current_trade.net_pnl)
                    daily_profit += max(0, current_trade.net_pnl)
                    trades_today.append(current_trade)
                    last_exit_bar_idx = bar_idx
                    current_trade = None
                    continue

            # --- Check for new entry ---
            if current_trade is None:
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

        # Force close at EOD
        if current_trade is not None and current_trade.is_open:
            self._close_trade(current_trade, day_data.iloc[-1], day_data.index[-1],
                              TradeStatus.TIME_EXIT)
            trades_today.append(current_trade)

        return trades_today

    def _generate_signal(self, row: pd.Series) -> Signal:
        """
        Generate entry signal:
        LONG:  Price > VWAP + Supertrend Green + Candle above 9 EMA
        SHORT: Price < VWAP + Supertrend Red  + Candle below 9 EMA
        """
        buy = row.get("buy_signal", False)
        sell = row.get("sell_signal", False)

        # Additional confirmation: prefer strong candles
        bullish = row.get("bullish_candle", False)
        bearish = row.get("bearish_candle", False)

        if buy and bullish:
            return Signal.LONG

        if sell and bearish:
            return Signal.SHORT

        # Allow entry even without strong candle if all 3 conditions align
        if buy:
            return Signal.LONG
        if sell:
            return Signal.SHORT

        return Signal.NONE

    def _enter_trade(self, row: pd.Series, ts, signal: Signal) -> Optional[Trade]:
        """Enter a scalping trade with SL below Supertrend."""
        price = row["close"]
        supertrend = row.get("supertrend", price)

        if pd.isna(supertrend) or supertrend <= 0:
            return None

        # Calculate SL based on Supertrend
        if signal == Signal.LONG:
            sl_price = supertrend - SL_BUFFER_POINTS
            sl_points = price - sl_price
        else:
            sl_price = supertrend + SL_BUFFER_POINTS
            sl_points = sl_price - price

        # Clamp SL
        sl_points = max(MIN_SL_POINTS, min(sl_points, MAX_SL_POINTS))

        if signal == Signal.LONG:
            sl_price = price - sl_points
        else:
            sl_price = price + sl_points

        # Check max SL in rupees
        max_sl_points = MAX_SL_PER_TRADE / (MAX_LOTS * NIFTY_LOT_SIZE)
        if sl_points > max_sl_points:
            sl_points = max_sl_points
            if signal == Signal.LONG:
                sl_price = price - sl_points
            else:
                sl_price = price + sl_points

        # Target based on R:R
        target_points = sl_points * RISK_REWARD_RATIO
        target_inr = target_points * MAX_LOTS * NIFTY_LOT_SIZE

        # Clamp target to ₹1,500–₹2,000 range
        if target_inr < TARGET_MIN:
            target_points = TARGET_MIN / (MAX_LOTS * NIFTY_LOT_SIZE)
        elif target_inr > TARGET_MAX:
            target_points = TARGET_MAX / (MAX_LOTS * NIFTY_LOT_SIZE)

        if signal == Signal.LONG:
            target_price = price + target_points
        else:
            target_price = price - target_points

        trade = Trade(
            entry_time=ts,
            direction=signal,
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

        return trade

    def _manage_trade(self, trade: Trade, row: pd.Series, ts, time_str: str) -> Trade:
        """Manage open trade: check SL, target, trailing, EOD, supertrend flip."""
        price = row["close"]
        high = row["high"]
        low = row["low"]

        # --- EOD Square-off ---
        if time_str >= SQUARE_OFF_TIME:
            self._close_trade(trade, row, ts, TradeStatus.TIME_EXIT)
            return trade

        if trade.direction == Signal.LONG:
            # --- Stop Loss ---
            if low <= trade.sl_price:
                trade.exit_price = trade.sl_price
                trade.exit_time = ts
                trade.status = TradeStatus.SL_HIT if not trade.trail_active else TradeStatus.TRAILING_SL
                self._calculate_pnl(trade)
                return trade

            # --- Target Hit ---
            if high >= trade.target_price:
                trade.exit_price = trade.target_price
                trade.exit_time = ts
                trade.status = TradeStatus.TARGET_HIT
                self._calculate_pnl(trade)
                return trade

            # Track max favorable
            trade.max_favorable = max(trade.max_favorable, high - trade.entry_price)

            # --- Supertrend flip to red = exit ---
            st_dir = row.get("supertrend_direction", 1)
            if st_dir == -1:
                self._close_trade(trade, row, ts, TradeStatus.TRAILING_SL)
                return trade

            # --- Trailing Stop ---
            profit_points = high - trade.entry_price
            if profit_points >= TRAIL_ACTIVATE_POINTS and not trade.trail_active:
                trade.trail_active = True
            if profit_points >= TIGHT_TRAIL_ACTIVATE and not trade.tight_trail_active:
                trade.tight_trail_active = True

            if trade.tight_trail_active:
                new_sl = high - profit_points * (1 - TIGHT_TRAIL_LOCK_PCT)
                trade.sl_price = max(trade.sl_price, new_sl)
            elif trade.trail_active:
                new_sl = high - profit_points * (1 - TRAIL_LOCK_PCT)
                trade.sl_price = max(trade.sl_price, new_sl)
            trade.trailing_sl = trade.sl_price

        else:  # SHORT
            if high >= trade.sl_price:
                trade.exit_price = trade.sl_price
                trade.exit_time = ts
                trade.status = TradeStatus.SL_HIT if not trade.trail_active else TradeStatus.TRAILING_SL
                self._calculate_pnl(trade)
                return trade

            if low <= trade.target_price:
                trade.exit_price = trade.target_price
                trade.exit_time = ts
                trade.status = TradeStatus.TARGET_HIT
                self._calculate_pnl(trade)
                return trade

            trade.max_favorable = max(trade.max_favorable, trade.entry_price - low)

            # Supertrend flip to green = exit
            st_dir = row.get("supertrend_direction", -1)
            if st_dir == 1:
                self._close_trade(trade, row, ts, TradeStatus.TRAILING_SL)
                return trade

            profit_points = trade.entry_price - low
            if profit_points >= TRAIL_ACTIVATE_POINTS and not trade.trail_active:
                trade.trail_active = True
            if profit_points >= TIGHT_TRAIL_ACTIVATE and not trade.tight_trail_active:
                trade.tight_trail_active = True

            if trade.tight_trail_active:
                new_sl = low + profit_points * (1 - TIGHT_TRAIL_LOCK_PCT)
                trade.sl_price = min(trade.sl_price, new_sl)
            elif trade.trail_active:
                new_sl = low + profit_points * (1 - TRAIL_LOCK_PCT)
                trade.sl_price = min(trade.sl_price, new_sl)
            trade.trailing_sl = trade.sl_price

        return trade

    def _close_trade(self, trade: Trade, row: pd.Series, ts, status: TradeStatus):
        """Close trade at current price."""
        trade.exit_price = row["close"]
        trade.exit_time = ts
        trade.status = status
        self._calculate_pnl(trade)

    def _calculate_pnl(self, trade: Trade):
        """Calculate trade P&L."""
        qty = trade.quantity

        if trade.direction == Signal.LONG:
            trade.gross_pnl = (trade.exit_price - trade.entry_price) * qty
        else:
            trade.gross_pnl = (trade.entry_price - trade.exit_price) * qty

        entry_costs = calculate_costs(trade.entry_price, qty, is_entry=True)
        exit_costs = calculate_costs(trade.exit_price, qty, is_entry=False)

        trade.costs = entry_costs["total"] + exit_costs["total"]
        trade.net_pnl = trade.gross_pnl - trade.costs

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

        # Target hit rate
        target_hits = sum(1 for t in self.trades if t.status == TradeStatus.TARGET_HIT)
        result.target_hit_rate = target_hits / len(self.trades) * 100

        # Avg trades per day
        if self.daily_pnl:
            unique_days = len(set(d["date"] for d in self.daily_pnl))
            result.avg_trades_per_day = len(self.trades) / max(1, unique_days)

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
        print("  🎯 SUPERTREND + VWAP SCALPING — BACKTEST RESULTS")
        print("═" * 70)

        trading_days = len(set(d["date"] for d in self.daily_pnl)) if self.daily_pnl else 0
        print(f"\n  {'Period:':<28} {trading_days} trading days")
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

        print(f"\n  {'─' * 45} Scalping {'─' * 16}")
        print(f"  {'Avg Trades/Day:':<28} {result.avg_trades_per_day:>12.1f}")
        print(f"  {'Target Hit Rate:':<28} {result.target_hit_rate:>11.1f}%")

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
