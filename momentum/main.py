"""
Dual-Confirmation Momentum Strategy — Main Runner
===================================================
Nifty Futures intraday momentum with dual MACD+RSI confirmation,
partial exits at 1:1 R:R, and dynamic trailing stops.

Usage:
    python main.py                    # Run backtest with sample data
    python main.py --days 180         # Custom period
    python main.py --no-charts        # Skip chart generation
    python main.py --dhan             # Use Dhan API for Nifty data

Output:  results/momentum_*.png, results/momentum_report.xlsx
"""

import argparse
import sys
import os
import time

# Ensure imports work
sys.path.insert(0, os.path.dirname(__file__))

from config import BACKTEST_DAYS, TIMEFRAME, INITIAL_CAPITAL
from data_utils import generate_nifty_data
from strategy import MomentumBacktest
from visualization import (
    plot_equity_curve, plot_monthly_heatmap,
    plot_trade_analysis, save_excel_report,
)


def main():
    parser = argparse.ArgumentParser(
        description="🚀 Nifty Dual-Confirmation Momentum Backtest"
    )
    parser.add_argument("--days", type=int, default=BACKTEST_DAYS,
                        help=f"Backtest period in days (default: {BACKTEST_DAYS})")
    parser.add_argument("--timeframe", type=str, default=TIMEFRAME,
                        help=f"Candle timeframe (default: {TIMEFRAME})")
    parser.add_argument("--no-charts", action="store_true",
                        help="Skip chart generation")
    parser.add_argument("--dhan", action="store_true",
                        help="Fetch data from Dhan API instead of sample data")
    parser.add_argument("--csv", type=str, default=None,
                        help="Load OHLCV data from CSV file")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  🚀 NIFTY DUAL-CONFIRMATION MOMENTUM")
    print("  MACD + RSI + Price Action | Partial Exits | Dynamic Trailing")
    print("=" * 70)

    # ── Load Data ──
    start_time = time.time()

    if args.csv:
        print(f"\n  Loading data from CSV: {args.csv}")
        import pandas as pd
        df = pd.read_csv(args.csv, parse_dates=["datetime"], index_col="datetime")
    elif args.dhan:
        print(f"\n  Fetching {args.days} days of Nifty data from Dhan API...")
        df = _fetch_dhan_data(args.days, args.timeframe)
    else:
        print(f"\n  Generating {args.days} days of sample Nifty data ({args.timeframe})...")
        df = generate_nifty_data(days=args.days, timeframe=args.timeframe)

    print(f"  Data loaded: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Nifty range: {df['close'].min():.0f} — {df['close'].max():.0f}")

    # ── Run Backtest ──
    engine = MomentumBacktest()
    result = engine.run(df)

    elapsed = time.time() - start_time

    # ── Print Summary ──
    engine.print_summary(result)
    print(f"\n  Backtest completed in {elapsed:.1f}s")

    # ── Generate Charts & Reports ──
    os.makedirs("results", exist_ok=True)

    if not args.no_charts:
        print("\n  Generating charts...")
        plot_equity_curve(result)
        plot_monthly_heatmap(result)
        plot_trade_analysis(result)

    print("\n  Generating Excel report...")
    try:
        save_excel_report(result)
    except PermissionError:
        alt_path = f"results/momentum_report_{int(time.time())}.xlsx"
        print(f"  ⚠ File locked, saving to: {alt_path}")
        save_excel_report(result, save_path=alt_path)

    print("\n" + "=" * 70)
    print("  All output saved to results/ folder")
    print("=" * 70 + "\n")

    return result


def _fetch_dhan_data(days: int, timeframe: str):
    """Fetch Nifty data from Dhan API in chunks (max 90 days per call)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backtest'))
    from dhan_fetch import fetch_nifty_intraday
    import pandas as pd
    from datetime import datetime, timedelta

    interval_map = {
        "1min": "1",
        "5min": "5",
        "15min": "15",
        "1H": "60",
    }
    interval = interval_map.get(timeframe, "5")

    all_frames = []
    chunk_days = 85
    end_date = datetime.now()
    remaining = days

    while remaining > 0:
        chunk = min(remaining, chunk_days)
        start = end_date - timedelta(days=chunk)

        print(f"    Fetching {start.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")

        df_chunk = fetch_nifty_intraday(
            from_date=start.strftime("%Y-%m-%d"),
            to_date=end_date.strftime("%Y-%m-%d"),
            interval=interval
        )

        if df_chunk is not None and not df_chunk.empty:
            all_frames.append(df_chunk)

        end_date = start - timedelta(days=1)
        remaining -= chunk

    if not all_frames:
        print("  ⚠ No data from Dhan API, falling back to sample data")
        return generate_nifty_data(days=days, timeframe=timeframe)

    df = pd.concat(all_frames).sort_index()
    df = df[~df.index.duplicated(keep='first')]

    return df


if __name__ == "__main__":
    main()
