"""
Nifty Sapphire Intraday — Main Runner
=======================================
Paired theta-focused short strangle with dynamic trailing SL.

Usage:
    python main.py                    # Run 1-year backtest with sample data
    python main.py --days 180         # Custom period
    python main.py --no-charts        # Skip chart generation
    python main.py --dhan             # Use Dhan API for Nifty data

Output:  results/sapphire_*.png, results/sapphire_report.xlsx
"""

import argparse
import sys
import os
import time

# Ensure imports work
sys.path.insert(0, os.path.dirname(__file__))

from config import BACKTEST_DAYS, TIMEFRAME, INITIAL_CAPITAL
from data_utils import generate_nifty_data
from strategy import SapphireBacktest
from visualization import (
    plot_equity_curve, plot_monthly_heatmap,
    plot_trade_analysis, save_excel_report,
)


def _save_trade_csv(result):
    """Save backtest trades to paper_trades CSV for tradebook."""
    import pandas as pd
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_trades")
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "sapphire_trade_log.csv")
    rows = []
    for i, t in enumerate(result.trades):
        date_val = t.date.date() if hasattr(t.date, 'date') and callable(t.date.date) else t.date
        rows.append({
            "trade_id": i + 1,
            "date": str(date_val),
            "entry_spot": round(t.entry_spot, 2),
            "exit_spot": round(t.spot_at_exit, 2) if t.spot_at_exit else 0,
            "ce_strike": t.ce_leg.strike if t.ce_leg else 0,
            "pe_strike": t.pe_leg.strike if t.pe_leg else 0,
            "ce_entry_prem": round(t.ce_leg.entry_premium, 2) if t.ce_leg else 0,
            "pe_entry_prem": round(t.pe_leg.entry_premium, 2) if t.pe_leg else 0,
            "ce_exit_prem": round(t.ce_leg.exit_premium, 2) if t.ce_leg else 0,
            "pe_exit_prem": round(t.pe_leg.exit_premium, 2) if t.pe_leg else 0,
            "gross_pnl": round(t.gross_pnl, 2),
            "costs": round(t.total_costs, 2),
            "net_pnl": round(t.net_pnl, 2),
            "exit_reason": t.exit_reason,
            "ce_status": t.ce_leg.status.value if t.ce_leg else "",
            "pe_status": t.pe_leg.status.value if t.pe_leg else "",
            "capital": 0,
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n  Trade log saved: {csv_path} ({len(rows)} trades)")


def main():
    parser = argparse.ArgumentParser(
        description="💎 Nifty Sapphire Intraday Short Strangle Backtest"
    )
    parser.add_argument("--days", type=int, default=BACKTEST_DAYS,
                        help=f"Backtest period in days (default: {BACKTEST_DAYS})")
    parser.add_argument("--timeframe", type=str, default=TIMEFRAME,
                        help=f"Candle timeframe (default: {TIMEFRAME})")
    parser.add_argument("--symbol", type=str, default="nifty",
                        choices=["nifty", "banknifty", "finnifty", "midcapnifty", "sensex"],
                        help="Index to trade (default: nifty)")
    parser.add_argument("--no-charts", action="store_true",
                        help="Skip chart generation")
    parser.add_argument("--csv", type=str, default=None,
                        help="Load OHLCV data from CSV file")
    
    args = parser.parse_args()
    _SYM_LABELS = {"nifty": "NIFTY", "banknifty": "BANK NIFTY", "finnifty": "FIN NIFTY",
                   "midcapnifty": "MIDCAP NIFTY", "sensex": "SENSEX"}
    sym_label = _SYM_LABELS.get(args.symbol, args.symbol.upper())
    
    print("\n" + "=" * 70)
    print(f"  💎 {sym_label} SAPPHIRE INTRADAY")
    print("  Paired Theta-Focused Short Strangle Strategy")
    print("=" * 70)
    
    # ── Load Data (real Dhan API data by default) ──
    start_time = time.time()
    
    if args.csv:
        print(f"\n  Loading data from CSV: {args.csv}")
        import pandas as pd
        df = pd.read_csv(args.csv, parse_dates=["datetime"], index_col="datetime")
    else:
        print(f"\n  Loading {args.days} days of {sym_label} data ({args.timeframe})...")
        df = generate_nifty_data(days=args.days, timeframe=args.timeframe,
                                 symbol=args.symbol)
    
    print(f"  Data loaded: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  {sym_label} range: {df['close'].min():.0f} — {df['close'].max():.0f}")
    
    # ── Run Backtest ──
    engine = SapphireBacktest()
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
        alt_path = f"results/sapphire_report_{int(time.time())}.xlsx"
        print(f"  ⚠ File locked, saving to: {alt_path}")
        save_excel_report(result, save_path=alt_path)
    
    print("\n" + "=" * 70)
    print("  All output saved to results/ folder")
    print("=" * 70 + "\n")
    
    return result


if __name__ == "__main__":
    main()
