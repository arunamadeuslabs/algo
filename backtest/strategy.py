"""
Core Strategy Engine: EMA Crossover + Option Selling Backtest
Implements the "Plus Sign" crossover system for Nifty option selling.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

from config import *
from data_utils import get_atm_strike, estimate_option_premium


def calculate_transaction_costs(entry_premium: float, exit_premium: float,
                                 quantity: int) -> dict:
    """
    Calculate all real-world transaction costs for an option trade.
    Returns breakdown dict + total cost.

    Costs for option SELLING (sell to enter, buy to exit):
      - Brokerage: ₹20 per order × 2 (entry + exit)
      - Slippage: SLIPPAGE_POINTS × quantity (both legs)
      - STT: 0.0625% on sell-side turnover only
      - Exchange txn charges: 0.05% on total turnover
      - SEBI charges: ₹10 per crore
      - Stamp duty: 0.003% on buy-side turnover
      - GST: 18% on (brokerage + exchange charges)
    """
    if not INCLUDE_COSTS:
        return {"total": 0, "brokerage": 0, "slippage": 0, "stt": 0,
                "exchange": 0, "gst": 0, "sebi": 0, "stamp": 0}

    sell_turnover = entry_premium * quantity  # Option seller enters by selling
    buy_turnover = exit_premium * quantity    # Exits by buying back
    total_turnover = sell_turnover + buy_turnover

    # Brokerage: flat per order
    brokerage = BROKERAGE_PER_ORDER * 2  # entry + exit

    # Slippage: adverse price movement on both legs
    slippage = SLIPPAGE_POINTS * quantity * 2  # entry gets lower, exit gets higher

    # STT: only on sell side for options
    stt = sell_turnover * STT_SELL_RATE

    # Exchange transaction charges
    exchange = total_turnover * EXCHANGE_TXN_RATE

    # SEBI charges
    sebi = total_turnover * SEBI_CHARGE_RATE

    # Stamp duty: buy side only
    stamp = buy_turnover * STAMP_DUTY_BUY_RATE

    # GST: 18% on brokerage + exchange charges
    gst = (brokerage + exchange) * GST_RATE

    total = brokerage + slippage + stt + exchange + gst + sebi + stamp

    return {
        "total": round(total, 2),
        "brokerage": round(brokerage, 2),
        "slippage": round(slippage, 2),
        "stt": round(stt, 2),
        "exchange": round(exchange, 2),
        "gst": round(gst, 2),
        "sebi": round(sebi, 2),
        "stamp": round(stamp, 2),
    }


class Signal(Enum):
    NONE = 0
    BUY = 1      # Bullish crossover → Sell PUT
    SELL = -1     # Bearish crossover → Sell CALL


class TradeStatus(Enum):
    OPEN = "OPEN"
    TARGET_HIT = "TARGET_HIT"
    SL_HIT = "SL_HIT"
    TRAILING_SL = "TRAILING_SL"
    TIME_EXIT = "TIME_EXIT"
    FORCED_EXIT = "FORCED_EXIT"


@dataclass
class Trade:
    entry_time: pd.Timestamp
    direction: Signal
    option_type: str          # "CE" or "PE"
    strike: float
    entry_spot: float
    entry_premium: float
    sl_spot: float            # SL in spot terms
    target_spot: float        # Target in spot terms
    sl_premium: float = 0.0
    target_premium: float = 0.0
    exit_time: Optional[pd.Timestamp] = None
    exit_premium: float = 0.0
    exit_spot: float = 0.0
    status: TradeStatus = TradeStatus.OPEN
    pnl: float = 0.0
    gross_pnl: float = 0.0
    costs: dict = field(default_factory=dict)  # Transaction cost breakdown
    total_costs: float = 0.0
    lots: int = NUM_LOTS
    trailing_sl: float = 0.0
    max_favorable: float = 0.0  # Max premium decay (profit for seller)

    @property
    def quantity(self):
        return self.lots * OPTION_LOT_SIZE

    def calculate_pnl(self):
        """PnL for option SELLER: profit = entry_premium - exit_premium - costs"""
        self.gross_pnl = (self.entry_premium - self.exit_premium) * self.quantity
        self.costs = calculate_transaction_costs(
            self.entry_premium, self.exit_premium, self.quantity
        )
        self.total_costs = self.costs["total"]
        self.pnl = self.gross_pnl - self.total_costs
        return self.pnl


@dataclass
class BacktestResult:
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    daily_pnl: dict = field(default_factory=dict)

    @property
    def total_trades(self):
        return len(self.trades)

    @property
    def winning_trades(self):
        return [t for t in self.trades if t.pnl > 0]

    @property
    def losing_trades(self):
        return [t for t in self.trades if t.pnl <= 0]

    @property
    def win_rate(self):
        if not self.trades:
            return 0
        return len(self.winning_trades) / len(self.trades) * 100

    @property
    def total_pnl(self):
        return sum(t.pnl for t in self.trades)

    @property
    def max_drawdown(self):
        if not self.equity_curve:
            return 0
        peak = self.equity_curve[0]
        max_dd = 0
        for val in self.equity_curve:
            peak = max(peak, val)
            dd = (peak - val) / peak * 100
            max_dd = max(max_dd, dd)
        return round(max_dd, 2)

    @property
    def avg_win(self):
        wins = [t.pnl for t in self.winning_trades]
        return np.mean(wins) if wins else 0

    @property
    def avg_loss(self):
        losses = [t.pnl for t in self.losing_trades]
        return np.mean(losses) if losses else 0

    @property
    def profit_factor(self):
        gross_profit = sum(t.pnl for t in self.winning_trades)
        gross_loss = abs(sum(t.pnl for t in self.losing_trades))
        return round(gross_profit / gross_loss, 2) if gross_loss > 0 else float('inf')

    @property
    def sharpe_ratio(self):
        if not self.trades:
            return 0
        returns = [t.pnl for t in self.trades]
        if np.std(returns) == 0:
            return 0
        return round(np.mean(returns) / np.std(returns) * np.sqrt(252), 2)

    def summary(self) -> dict:
        return {
            "Total Trades": self.total_trades,
            "Winning Trades": len(self.winning_trades),
            "Losing Trades": len(self.losing_trades),
            "Win Rate (%)": round(self.win_rate, 2),
            "Total PnL (₹)": round(self.total_pnl, 2),
            "Avg Win (₹)": round(self.avg_win, 2),
            "Avg Loss (₹)": round(self.avg_loss, 2),
            "Profit Factor": self.profit_factor,
            "Max Drawdown (%)": self.max_drawdown,
            "Sharpe Ratio": self.sharpe_ratio,
        }


class OptionSellingBacktest:
    """
    Backtest engine for EMA Crossover Option Selling Strategy.

    Logic:
    - Bullish crossover (EMA9 > EMA21 + close > EMA21) → SELL PUT (collect premium)
    - Bearish crossover (EMA9 < EMA21 + close < EMA21) → SELL CALL (collect premium)

    Option seller profits when:
    - Premium decays (time decay / theta)
    - Spot moves in favorable direction
    """

    def __init__(self, data: pd.DataFrame, capital: float = INITIAL_CAPITAL):
        self.data = data
        self.initial_capital = capital
        self.capital = capital
        self.current_trade: Optional[Trade] = None
        self.result = BacktestResult()
        self.daily_loss = 0.0
        self.current_date = None
        # Signal diagnostics
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

    def _generate_signal(self, idx: int) -> Signal:
        """
        Generate trading signal based on EMA crossover + confirmations.
        This is where the "Plus Sign" logic is implemented.
        """
        row = self.data.iloc[idx]
        if idx < 1:
            return Signal.NONE

        # --- Core Signal: EMA Crossover ---
        is_bull_cross = row.get("crossover_bull", False)
        is_bear_cross = row.get("crossover_bear", False)

        if not (is_bull_cross or is_bear_cross):
            return Signal.NONE

        self.diag["total_crossovers"] += 1

        # --- Filter 1: Candle close vs MAs ---
        close = row["close"]
        ema_fast = row["ema_fast"]
        ema_slow = row["ema_slow"]

        if is_bull_cross and close <= ema_slow:
            self.diag["filtered_close_vs_ma"] += 1
            return Signal.NONE  # Close must be above both MAs for buy
        if is_bear_cross and close >= ema_slow:
            self.diag["filtered_close_vs_ma"] += 1
            return Signal.NONE  # Close must be below both MAs for sell

        # --- Filter 2: Volume confirmation ---
        if not row.get("high_volume", False):
            self.diag["filtered_volume"] += 1
            return Signal.NONE

        # --- Filter 3: Not sideways (ADX filter) ---
        if not row.get("not_sideways", True):
            self.diag["filtered_adx"] += 1
            return Signal.NONE

        # --- Filter 4: Time filter ---
        bar_time = self.data.index[idx]
        if hasattr(bar_time, 'hour'):
            time_str = f"{bar_time.hour:02d}:{bar_time.minute:02d}"
            if time_str < TRADING_START or time_str > TRADING_END:
                self.diag["filtered_time"] += 1
                return Signal.NONE

        self.diag["signals_passed"] += 1

        if is_bull_cross:
            return Signal.BUY
        elif is_bear_cross:
            return Signal.SELL

        return Signal.NONE

    def _enter_trade(self, idx: int, signal: Signal):
        """
        Enter an option selling trade on the NEXT candle after signal.
        BUY signal → Sell PUT (bullish view)
        SELL signal → Sell CALL (bearish view)
        """
        if idx + 1 >= len(self.data):
            return

        entry_bar = self.data.iloc[idx + 1]  # Enter on NEXT candle
        entry_time = self.data.index[idx + 1]
        spot = entry_bar["close"]

        # Determine option type
        if signal == Signal.BUY:
            option_type = "PE"  # Sell PUT on bullish signal
            strike = get_atm_strike(spot)
            sl_spot = min(entry_bar.get("recent_swing_low", spot - STOP_LOSS_POINTS),
                          entry_bar["ema_slow"]) - 10
            target_spot = spot + TARGET_POINTS
        else:
            option_type = "CE"  # Sell CALL on bearish signal
            strike = get_atm_strike(spot)
            sl_spot = max(entry_bar.get("recent_swing_high", spot + STOP_LOSS_POINTS),
                          entry_bar["ema_slow"]) + 10
            target_spot = spot - TARGET_POINTS

        # Estimate premium
        dte = DAYS_TO_EXPIRY_MAX
        entry_premium = estimate_option_premium(spot, strike, dte, option_type)

        if entry_premium < PREMIUM_COLLECTION_MIN * 0.5:
            self.diag["premium_too_low"] += 1
            return  # Premium too low, skip

        # --- Premium-based SL/Target for 1:3 RR ---
        if USE_PREMIUM_BASED_EXIT:
            sl_premium = entry_premium * (1 + SL_PREMIUM_PCT)  # SL: premium rises 30%
            # Target: premium drops. Risk = SL_PREMIUM_PCT * entry.
            # Reward = RR * Risk = 3 * 30% * entry = 90% drop from entry
            target_premium = max(5.0, entry_premium * (1 - SL_PREMIUM_PCT * RISK_REWARD_RATIO))
        else:
            sl_premium = 0.0
            target_premium = 0.0

        trade = Trade(
            entry_time=entry_time,
            direction=signal,
            option_type=option_type,
            strike=strike,
            entry_spot=spot,
            entry_premium=entry_premium,
            sl_spot=sl_spot,
            target_spot=target_spot,
            sl_premium=sl_premium,
            target_premium=target_premium,
            lots=NUM_LOTS,
            trailing_sl=sl_spot,
        )

        self.current_trade = trade

    def _manage_trade(self, idx: int):
        """
        Manage open trade: check SL, target, trailing stop, time exit.
        """
        if self.current_trade is None:
            return

        trade = self.current_trade
        row = self.data.iloc[idx]
        bar_time = self.data.index[idx]
        spot = row["close"]
        high = row["high"]
        low = row["low"]

        # Estimate current premium
        bars_elapsed = (bar_time - trade.entry_time).total_seconds() / 3600
        dte_remaining = max(0.1, DAYS_TO_EXPIRY_MAX - bars_elapsed / 6.5)
        current_premium = estimate_option_premium(spot, trade.strike,
                                                   int(dte_remaining),
                                                   trade.option_type)

        # Track max favorable excursion
        premium_decay = trade.entry_premium - current_premium
        trade.max_favorable = max(trade.max_favorable, premium_decay)

        # === Premium-based SL/Target (1:3 RR) ===
        if USE_PREMIUM_BASED_EXIT:
            # SL: Premium rises above SL level (going against seller)
            if current_premium >= trade.sl_premium:
                self._exit_trade(idx, current_premium, TradeStatus.SL_HIT)
                return

            # Target: Premium drops below target level (theta decay in seller's favor)
            if current_premium <= trade.target_premium:
                self._exit_trade(idx, current_premium, TradeStatus.TARGET_HIT)
                return
        else:
            # --- Spot-based Stop Loss ---
            if trade.direction == Signal.BUY:  # Sold PUT
                if low <= trade.trailing_sl:
                    self._exit_trade(idx, current_premium, TradeStatus.SL_HIT)
                    return
            else:  # Sold CALL
                if high >= trade.trailing_sl:
                    self._exit_trade(idx, current_premium, TradeStatus.SL_HIT)
                    return

            # --- Spot-based Target ---
            if trade.direction == Signal.BUY:
                if spot >= trade.target_spot:
                    self._exit_trade(idx, current_premium, TradeStatus.TARGET_HIT)
                    return
            else:
                if spot <= trade.target_spot:
                    self._exit_trade(idx, current_premium, TradeStatus.TARGET_HIT)
                    return

        # --- Trailing Stop (using Slow EMA) --- only when NOT using premium exits ---
        if TRAILING_STOP_USE_EMA and not USE_PREMIUM_BASED_EXIT:
            ema_slow = row["ema_slow"]
            if trade.direction == Signal.BUY:
                new_trailing = max(trade.trailing_sl, ema_slow - 10)
                trade.trailing_sl = new_trailing
            else:
                new_trailing = min(trade.trailing_sl, ema_slow + 10)
                if trade.trailing_sl == 0 or new_trailing < trade.trailing_sl:
                    trade.trailing_sl = new_trailing

        # --- Time-based exit (end of day) ---
        if hasattr(bar_time, 'hour'):
            time_str = f"{bar_time.hour:02d}:{bar_time.minute:02d}"
            if time_str >= SQUARE_OFF_TIME:
                self._exit_trade(idx, current_premium, TradeStatus.TIME_EXIT)
                return

        # --- Max daily loss check ---
        unrealized_pnl = (trade.entry_premium - current_premium) * trade.quantity
        if self.daily_loss + min(0, unrealized_pnl) < -MAX_LOSS_PER_DAY:
            self._exit_trade(idx, current_premium, TradeStatus.FORCED_EXIT)
            return

    def _exit_trade(self, idx: int, exit_premium: float, status: TradeStatus):
        """Close the current trade and record results."""
        if self.current_trade is None:
            return

        trade = self.current_trade
        trade.exit_time = self.data.index[idx]
        trade.exit_spot = self.data.iloc[idx]["close"]
        trade.exit_premium = exit_premium
        trade.status = status
        trade.calculate_pnl()

        self.capital += trade.pnl
        self.daily_loss += min(0, trade.pnl)
        self.result.trades.append(trade)
        self.current_trade = None

    def run(self) -> BacktestResult:
        """
        Run the full backtest.
        """
        print("=" * 60)
        print("  NIFTY OPTION SELLING BACKTEST")
        print("  Strategy: EMA Crossover + Plus Sign System")
        print(f"  EMA: {FAST_EMA_PERIOD} / {SLOW_EMA_PERIOD}")
        print(f"  Capital: ₹{self.initial_capital:,.0f}")
        print(f"  Timeframe: {TIMEFRAME}")
        print(f"  Data bars: {len(self.data)}")
        print("=" * 60)

        pending_signal = Signal.NONE

        for idx in range(len(self.data)):
            bar_time = self.data.index[idx]

            # Reset daily loss tracker
            current_date = bar_time.date() if hasattr(bar_time, 'date') else bar_time
            if current_date != self.current_date:
                self.current_date = current_date
                self.daily_loss = 0.0

            # Manage existing trade
            if self.current_trade is not None:
                self._manage_trade(idx)
                # Check if a signal fires while we're in a trade
                signal = self._generate_signal(idx)
                if signal != Signal.NONE:
                    self.diag["blocked_by_open_trade"] += 1
            else:
                # If we had a pending signal from previous bar, enter now
                if pending_signal != Signal.NONE:
                    self._enter_trade(idx - 1, pending_signal)
                    pending_signal = Signal.NONE
                else:
                    # Generate new signal
                    signal = self._generate_signal(idx)
                    if signal != Signal.NONE and self.current_trade is None:
                        # Don't enter immediately — wait for next candle
                        pending_signal = signal

            # Track equity
            unrealized = 0
            if self.current_trade:
                dte_est = max(0.1, DAYS_TO_EXPIRY_MAX - 1)
                curr_prem = estimate_option_premium(
                    self.data.iloc[idx]["close"],
                    self.current_trade.strike,
                    int(dte_est),
                    self.current_trade.option_type
                )
                unrealized = (self.current_trade.entry_premium - curr_prem) * self.current_trade.quantity

            self.result.equity_curve.append(self.capital + unrealized)

        # Force close any open trade at end
        if self.current_trade is not None:
            last_prem = estimate_option_premium(
                self.data.iloc[-1]["close"],
                self.current_trade.strike, 1,
                self.current_trade.option_type
            )
            self._exit_trade(len(self.data) - 1, last_prem, TradeStatus.TIME_EXIT)

        self.diag["trades_entered"] = len(self.result.trades)
        self._print_diagnostics()
        return self.result

    def _print_diagnostics(self):
        """Print signal filtering diagnostics."""
        d = self.diag
        print("\n" + "─" * 50)
        print("  🔍 SIGNAL FILTER DIAGNOSTICS")
        print("─" * 50)
        print(f"  Total EMA crossovers detected:  {d['total_crossovers']}")
        print(f"  ❌ Filtered: Close vs MA:        {d['filtered_close_vs_ma']}")
        print(f"  ❌ Filtered: Low volume:          {d['filtered_volume']}")
        print(f"  ❌ Filtered: Sideways (ADX<20):   {d['filtered_adx']}")
        print(f"  ❌ Filtered: Outside trading hrs:  {d['filtered_time']}")
        print(f"  ✅ Signals that passed filters:   {d['signals_passed']}")
        print(f"  ⏳ Blocked (already in trade):    {d['blocked_by_open_trade']}")
        print(f"  ⚠️  Premium too low:              {d['premium_too_low']}")
        print(f"  ✅ Trades actually entered:       {d['trades_entered']}")
        print("─" * 50)
