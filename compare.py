"""
Index Comparison Runner
========================
Run all 4 strategies on both Nifty and Bank Nifty to find which index
gives better P&L per strategy.

Usage:
    python compare.py                    # Compare both indices, all strategies
    python compare.py --strategies ema ironcondor   # Specific strategies
    python compare.py --days 180         # Custom period
"""

import subprocess
import sys
import os
import re
import argparse
from pathlib import Path

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PYTHON = sys.executable

STRATEGIES = {
    "ema":        {"dir": "backtest",    "script": "main.py", "default_symbol": "nifty"},
    "sapphire":   {"dir": "sapphire",    "script": "main.py", "default_symbol": "nifty"},
    "momentum":   {"dir": "momentum",    "script": "main.py", "default_symbol": "nifty"},
    "ironcondor": {"dir": "ironcondor",  "script": "main.py", "default_symbol": "banknifty"},
}

SYMBOLS = ["nifty", "banknifty", "finnifty", "midcapnifty", "sensex"]


def run_backtest(strategy: str, symbol: str, days: int) -> dict:
    """Run a single backtest and parse the output for key metrics."""
    info = STRATEGIES[strategy]
    cwd = str(BASE_DIR / info["dir"])
    cmd = [PYTHON, info["script"], "--no-charts", "--symbol", symbol, "--days", str(days)]

    result = subprocess.run(
        cmd, cwd=cwd,
        capture_output=True, text=True, timeout=120,
        encoding="utf-8", errors="replace",
    )

    output = result.stdout + result.stderr

    # Parse key metrics from output
    metrics = {
        "strategy": strategy,
        "symbol": symbol.upper(),
        "roi": _extract(output, r"ROI:\s*([\-+]?\d+\.?\d*)%"),
        "net_pnl": _extract_currency(output, r"Net P(?:&|n)L:\s*(?:Rs\s*)?[₹]?\s*([\-+]?[\d,]+)"),
        "trades": _extract_int(output, r"Total Trades:\s*(\d+)"),
        "win_rate": _extract(output, r"Winners:.*?(\d+\.?\d*)%"),
        "sharpe": _extract(output, r"Sharpe Ratio:\s*([\-+]?\d+\.?\d*)"),
        "max_dd": _extract(output, r"Max Drawdown:\s*([\-+]?\d+\.?\d*)%"),
        "profit_factor": _extract(output, r"Profit Factor:\s*([\-+]?\d+\.?\d*)"),
    }

    return metrics


def _extract(text: str, pattern: str) -> float:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else 0.0

def _extract_int(text: str, pattern: str) -> int:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else 0

def _extract_currency(text: str, pattern: str) -> float:
    m = re.search(pattern, text)
    if m:
        return float(m.group(1).replace(",", ""))
    return 0.0


def print_comparison(results: list):
    """Print side-by-side comparison table."""
    print("\n" + "=" * 95)
    print("  INDEX COMPARISON — WHICH INDEX WORKS BEST PER STRATEGY?")
    print("=" * 95)

    # Group by strategy
    from collections import defaultdict
    by_strat = defaultdict(list)
    for r in results:
        by_strat[r["strategy"]].append(r)

    # Header
    print(f"\n  {'Strategy':<14} {'Symbol':<12} {'ROI':>8} {'Trades':>7} {'Win%':>7} "
          f"{'Sharpe':>8} {'MaxDD':>8} {'PF':>6} {'Net PnL':>12}")
    print(f"  {'─' * 90}")

    best_overall = None
    best_roi = -999

    for strat in STRATEGIES:
        runs = by_strat.get(strat, [])
        for r in runs:
            roi_str = f"{r['roi']:+.1f}%"
            wr_str = f"{r['win_rate']:.1f}%"
            sharpe_str = f"{r['sharpe']:.2f}"
            dd_str = f"{r['max_dd']:.1f}%"
            pf_str = f"{r['profit_factor']:.2f}"
            pnl_str = f"Rs {r['net_pnl']:+,.0f}"

            # Highlight the better symbol for this strategy
            marker = ""
            if len(runs) == 2:
                other = [x for x in runs if x["symbol"] != r["symbol"]]
                if other and r["roi"] > other[0]["roi"]:
                    marker = " ★"

            print(f"  {strat:<14} {r['symbol']:<12} {roi_str:>8} {r['trades']:>7} "
                  f"{wr_str:>7} {sharpe_str:>8} {dd_str:>8} {pf_str:>6} {pnl_str:>12}{marker}")

            if r["roi"] > best_roi:
                best_roi = r["roi"]
                best_overall = r

        if len(runs) == 2:
            print(f"  {'─' * 90}")

    # Summary
    if best_overall:
        print(f"\n  ★ BEST COMBINATION: {best_overall['strategy'].upper()} on "
              f"{best_overall['symbol']} → {best_overall['roi']:+.1f}% ROI, "
              f"Sharpe {best_overall['sharpe']:.2f}")

    print("=" * 95 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare Nifty vs Bank Nifty across strategies")
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES.keys()),
                        choices=list(STRATEGIES.keys()),
                        help="Strategies to compare (default: all)")
    parser.add_argument("--days", type=int, default=90,
                        help="Backtest period in days (default: 90)")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS,
                        choices=SYMBOLS,
                        help="Indices to compare (default: both)")

    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  NIFTY vs BANK NIFTY — STRATEGY COMPARISON")
    print(f"  Period: {args.days} days | Strategies: {', '.join(args.strategies)}")
    print("=" * 65)

    results = []
    total = len(args.strategies) * len(args.symbols)
    count = 0

    for strat in args.strategies:
        for symbol in args.symbols:
            count += 1
            label = symbol.upper()
            print(f"\n  [{count}/{total}] Running {strat.upper()} on {label}...")

            try:
                metrics = run_backtest(strat, symbol, args.days)
                results.append(metrics)
                print(f"         → ROI: {metrics['roi']:+.1f}%, "
                      f"Win: {metrics['win_rate']:.0f}%, "
                      f"Sharpe: {metrics['sharpe']:.2f}")
            except subprocess.TimeoutExpired:
                print(f"         → TIMEOUT (skipped)")
            except Exception as e:
                print(f"         → ERROR: {e}")

    if results:
        print_comparison(results)


if __name__ == "__main__":
    main()
