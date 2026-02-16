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
from dhan_fetch import fetch_nifty_intraday, fetch_nifty_daily, save_data_to_csv


def main():
    parser = argparse.ArgumentParser(
        description="Nifty Option Selling Backtest - EMA Crossover Strategy"
    )
    parser.add_argument("--csv", type=str, help="Path to CSV file with OHLCV data")
    parser.add_argument("--dhan", action="store_true",
                        help="Fetch real data from Dhan API")
    parser.add_argument("--days", type=int, default=30,
                        help="Days of data (default: 30, max 90 for intraday)")
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

    print("\n" + "=" * 60)
    print("  🚀 NIFTY OPTION SELLING BACKTEST ENGINE")
    print("  Strategy: EMA Crossover + Plus Sign System")
    print("=" * 60)

    # --- Load Data ---
    if args.dhan:
        # Fetch real data from Dhan API
        interval_map = {"1min": 1, "5min": 5, "15min": 15, "1H": 60, "4H": 60, "Daily": 0}
        interval = interval_map.get(args.timeframe, 5)

        if args.timeframe == "Daily":
            print(f"\n📡 Fetching real Nifty DAILY data from Dhan API ({args.days} days)...")
            data = fetch_nifty_daily(days_back=args.days)
        else:
            days = min(args.days, 90)  # Dhan intraday limit
            print(f"\n📡 Fetching real Nifty {args.timeframe} data from Dhan API ({days} days)...")
            data = fetch_nifty_intraday(interval=interval, days_back=days)

        if data.empty:
            print("\n  ❌ No data from Dhan API. Falling back to sample data.")
            data = generate_sample_nifty_data(days=args.days, timeframe=args.timeframe)
        else:
            # Save fetched data for future use
            csv_name = f"nifty_{args.timeframe}_{args.days}d_dhan.csv"
            save_data_to_csv(data, csv_name)

    elif args.csv:
        print(f"\n📂 Loading data from: {args.csv}")
        data = load_csv_data(args.csv)
    else:
        print(f"\n📊 Generating {args.days} days of sample Nifty data ({args.timeframe})...")
        data = generate_sample_nifty_data(days=args.days, timeframe=args.timeframe)

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
