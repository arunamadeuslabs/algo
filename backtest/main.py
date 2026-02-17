"""
Nifty Option Selling Backtest - Main Runner
============================================
Strategy: EMA 9/21 Crossover ("Plus Sign" System)

Signal Logic:
  BUY  → EMA9 crosses above EMA21, close > EMA21, volume up → SELL PUT
  SELL → EMA9 crosses below EMA21, close < EMA21, volume up → SELL CALL

Filters:
  - Volume must be above 20-SMA * 1.2
  - ADX > 20 (trending market, avoid sideways)
  - Trade only between 09:20 - 15:15 IST
  - Enter on NEXT candle after signal (not signal candle)

Risk:
  - Fixed SL / Trailing with Slow EMA
  - Target: 1:2 minimum R:R
  - Max daily loss capped
  - Auto square-off at 15:25

Usage:
  python main.py                    # Run with sample data
  python main.py --csv data.csv     # Run with your CSV data
  python main.py --days 180         # 180 days sample data
"""

import sys
import os
import argparse

# Add backtest directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from data_utils import generate_sample_nifty_data, load_csv_data, compute_indicators
from strategy import OptionSellingBacktest
from visualization import generate_report


def _save_trade_csv(result):
    """Save backtest trades to paper_trades CSV for tradebook."""
    import pandas as pd
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_trades")
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "paper_trade_log.csv")
    rows = []
    for i, t in enumerate(result.trades):
        rows.append({
            "id": i + 1,
            "entry_time": str(t.entry_time),
            "direction": t.direction.name,
            "option_type": t.option_type,
            "strike": t.strike,
            "entry_spot": round(t.entry_spot, 2),
            "entry_premium": round(t.entry_premium, 2),
            "sl_premium": round(t.sl_premium, 2),
            "target_premium": round(t.target_premium, 2),
            "sl_spot": round(t.sl_spot, 2),
            "target_spot": round(t.target_spot, 2),
            "lots": t.lots,
            "status": t.status.value,
            "exit_time": str(t.exit_time) if t.exit_time else "",
            "exit_spot": round(t.exit_spot, 2),
            "exit_premium": round(t.exit_premium, 2),
            "gross_pnl": round(t.gross_pnl, 2),
            "costs": round(t.total_costs, 2),
            "net_pnl": round(t.pnl, 2),
            "max_favorable": round(t.max_favorable, 2),
            "quantity": t.quantity,
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n  Trade log saved: {csv_path} ({len(rows)} trades)")


def main():
    parser = argparse.ArgumentParser(
        description="Nifty Option Selling Backtest - EMA Crossover Strategy"
    )
    parser.add_argument("--csv", type=str, help="Path to CSV file with OHLCV data")
    parser.add_argument("--days", type=int, default=90,
                        help="Days of data (default: 90)")
    parser.add_argument("--symbol", type=str, default="nifty",
                        choices=["nifty", "banknifty", "finnifty", "midcapnifty", "sensex"],
                        help="Index to trade (default: nifty)")
    parser.add_argument("--capital", type=float, default=INITIAL_CAPITAL,
                        help=f"Starting capital (default: {INITIAL_CAPITAL})")
    parser.add_argument("--timeframe", type=str, default=TIMEFRAME,
                        choices=["1min", "5min", "15min", "1H", "4H", "Daily"],
                        help=f"Timeframe (default: {TIMEFRAME})")
    parser.add_argument("--fast-ema", type=int, default=FAST_EMA_PERIOD,
                        help=f"Fast EMA period (default: {FAST_EMA_PERIOD})")
    parser.add_argument("--slow-ema", type=int, default=SLOW_EMA_PERIOD,
                        help=f"Slow EMA period (default: {SLOW_EMA_PERIOD})")
    parser.add_argument("--no-charts", action="store_true", help="Skip chart generation")

    args = parser.parse_args()
    _SYM_LABELS = {"nifty": "NIFTY", "banknifty": "BANK NIFTY", "finnifty": "FIN NIFTY",
                   "midcapnifty": "MIDCAP NIFTY", "sensex": "SENSEX"}
    sym_label = _SYM_LABELS.get(args.symbol, args.symbol.upper())

    print("\n" + "=" * 60)
    print(f"  🚀 {sym_label} OPTION SELLING BACKTEST ENGINE")
    print("  Strategy: EMA Crossover + Plus Sign System")
    print("=" * 60)

    # --- Load Data (real Dhan API data by default) ---
    if args.csv:
        print(f"\n📂 Loading data from: {args.csv}")
        data = load_csv_data(args.csv)
    else:
        print(f"\n📊 Loading {args.days} days of {sym_label} data ({args.timeframe})...")
        data = generate_sample_nifty_data(days=args.days, timeframe=args.timeframe,
                                          symbol=args.symbol)

    print(f"   Bars loaded: {len(data)}")
    print(f"   Date range: {data.index[0]} → {data.index[-1]}")
    print(f"   Price range: {data['close'].min():.0f} - {data['close'].max():.0f}")

    # --- Compute Indicators ---
    print(f"\n📐 Computing indicators (EMA {args.fast_ema}/{args.slow_ema}, ADX, RSI, Volume)...")
    data = compute_indicators(data, fast_period=args.fast_ema, slow_period=args.slow_ema)

    bull_signals = data['crossover_bull'].sum()
    bear_signals = data['crossover_bear'].sum()
    print(f"   Bullish crossovers: {bull_signals}")
    print(f"   Bearish crossovers: {bear_signals}")
    print(f"   Trending bars (ADX>20): {data['not_sideways'].sum()} / {len(data)}")

    # --- Run Backtest ---
    print(f"\n⚡ Running backtest...")
    engine = OptionSellingBacktest(data, capital=args.capital)
    result = engine.run()

    # --- Save trade log CSV ---
    _save_trade_csv(result)

    # --- Generate Report ---
    if not args.no_charts:
        generate_report(data, result, args.capital)
    else:
        from visualization import print_summary_report
        print_summary_report(result, args.capital)

    # --- Return summary for programmatic use ---
    return result.summary()


if __name__ == "__main__":
    summary = main()
