"""
Bank Nifty Iron Condor — Main Runner
=======================================
Sell OTM Call Spread + OTM Put Spread with range-bound filters.

Usage:
    python main.py                    # Run 1-year backtest with sample data
    python main.py --days 180         # Custom period
    python main.py --no-charts        # Skip chart generation
    python main.py --dhan             # Use Dhan API for Bank Nifty data

Output:  results/ic_*.png, results/ic_report.xlsx
"""

import argparse
import sys
import os
import time

# Ensure imports work
sys.path.insert(0, os.path.dirname(__file__))

from config import BACKTEST_DAYS, TIMEFRAME, INITIAL_CAPITAL
from data_utils import generate_banknifty_data
from strategy import IronCondorBacktest
from visualization import (
    plot_equity_curve, plot_monthly_heatmap,
    plot_trade_analysis, save_excel_report,
)


def _save_trade_csv(result):
    """Save backtest trades to paper_trades CSV for tradebook."""
    import pandas as pd
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_trades")
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "ic_trade_log.csv")
    rows = []
    for i, t in enumerate(result.trades):
        date_val = t.date.date() if hasattr(t.date, 'date') and callable(t.date.date) else t.date
        rows.append({
            "trade_id": i + 1,
            "date": str(date_val),
            "entry_spot": round(t.entry_spot, 2),
            "exit_spot": round(t.spot_at_exit, 2) if t.spot_at_exit else 0,
            "sell_ce_strike": t.short_ce.strike if t.short_ce else 0,
            "buy_ce_strike": t.long_ce.strike if t.long_ce else 0,
            "sell_pe_strike": t.short_pe.strike if t.short_pe else 0,
            "buy_pe_strike": t.long_pe.strike if t.long_pe else 0,
            "sell_ce_entry_prem": round(t.short_ce.entry_premium, 2) if t.short_ce else 0,
            "buy_ce_entry_prem": round(t.long_ce.entry_premium, 2) if t.long_ce else 0,
            "sell_pe_entry_prem": round(t.short_pe.entry_premium, 2) if t.short_pe else 0,
            "buy_pe_entry_prem": round(t.long_pe.entry_premium, 2) if t.long_pe else 0,
            "sell_ce_exit_prem": round(t.short_ce.exit_premium, 2) if t.short_ce else 0,
            "buy_ce_exit_prem": round(t.long_ce.exit_premium, 2) if t.long_ce else 0,
            "sell_pe_exit_prem": round(t.short_pe.exit_premium, 2) if t.short_pe else 0,
            "buy_pe_exit_prem": round(t.long_pe.exit_premium, 2) if t.long_pe else 0,
            "net_credit": round(t.total_net_credit, 2),
            "gross_pnl": round(t.gross_pnl, 2),
            "costs": round(t.total_costs, 2),
            "net_pnl": round(t.net_pnl, 2),
            "exit_reason": t.exit_reason,
            "vix": round(t.vix_at_entry, 2),
            "rsi": round(t.rsi_at_entry, 2),
            "capital": 0,
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n  Trade log saved: {csv_path} ({len(rows)} trades)")


def main():
    parser = argparse.ArgumentParser(
        description="Bank Nifty Iron Condor Backtest"
    )
    parser.add_argument("--days", type=int, default=BACKTEST_DAYS,
                        help=f"Backtest period in days (default: {BACKTEST_DAYS})")
    parser.add_argument("--symbol", type=str, default="banknifty",
                        choices=["nifty", "banknifty", "finnifty", "midcapnifty", "sensex"],
                        help="Index to trade (default: banknifty)")
    parser.add_argument("--timeframe", type=str, default=TIMEFRAME,
                        help=f"Candle timeframe (default: {TIMEFRAME})")
    parser.add_argument("--no-charts", action="store_true",
                        help="Skip chart generation")
    parser.add_argument("--csv", type=str, default=None,
                        help="Load OHLCV data from CSV file")

    args = parser.parse_args()
    _SYM_LABELS = {"nifty": "NIFTY", "banknifty": "BANK NIFTY", "finnifty": "FIN NIFTY",
                   "midcapnifty": "MIDCAP NIFTY", "sensex": "SENSEX"}
    sym_label = _SYM_LABELS.get(args.symbol, args.symbol.upper())

    print("\n" + "=" * 70)
    print(f"  {sym_label} IRON CONDOR")
    print("  OTM Call Spread + OTM Put Spread — Defined Risk Credit Strategy")
    print("=" * 70)

    # ── Load Data (real Dhan API data by default) ──
    start_time = time.time()

    if args.csv:
        print(f"\n  Loading data from CSV: {args.csv}")
        import pandas as pd
        df = pd.read_csv(args.csv, parse_dates=["datetime"], index_col="datetime")
    else:
        print(f"\n  Loading {args.days} days of {sym_label} data ({args.timeframe})...")
        df = generate_banknifty_data(days=args.days, timeframe=args.timeframe,
                                     symbol=args.symbol)

    print(f"  Data loaded: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  {sym_label} range: {df['close'].min():.0f} — {df['close'].max():.0f}")

    # ── Run Backtest ──
    engine = IronCondorBacktest()
    result = engine.run(df)

    # ── Save trade log CSV ──
    _save_trade_csv(result)

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
        alt_path = f"results/ic_report_{int(time.time())}.xlsx"
        print(f"  Warning: File locked, saving to: {alt_path}")
        save_excel_report(result, save_path=alt_path)

    print("\n" + "=" * 70)
    print("  All output saved to results/ folder")
    print("=" * 70 + "\n")

    return result


if __name__ == "__main__":
    main()
