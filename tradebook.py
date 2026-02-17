"""
Trade Book Generator
=====================
Generates a comprehensive HTML trade book page showing ALL trades
across all strategies (EMA, Sapphire, Momentum, Supertrend).

Features:
  - Strategy tabs with color coding
  - Sortable columns (click headers)
  - Date range filter
  - P&L stats per strategy
  - Running cumulative P&L
  - Responsive dark-themed UI matching dashboard.html

Usage:
  python tradebook.py                 # Generate tradebook.html
  python tradebook.py --open          # Generate and open in browser
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

# ── Paths ────────────────────────────────────────────────────
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
TRADEBOOK_FILE = BASE_DIR / "tradebook.html"

EMA_TRADE_LOG = BASE_DIR / "backtest" / "paper_trades" / "paper_trade_log.csv"
EMA_STATE = BASE_DIR / "backtest" / "paper_trades" / "state.json"

SAPPHIRE_TRADE_LOG = BASE_DIR / "sapphire" / "paper_trades" / "sapphire_trade_log.csv"
SAPPHIRE_STATE = BASE_DIR / "sapphire" / "paper_trades" / "sapphire_state.json"

MOMENTUM_TRADE_LOG = BASE_DIR / "momentum" / "paper_trades" / "momentum_trade_log.csv"
MOMENTUM_STATE = BASE_DIR / "momentum" / "paper_trades" / "momentum_state.json"

SUPERTREND_TRADE_LOG = BASE_DIR / "supertrend" / "paper_trades" / "supertrend_trade_log.csv"
SUPERTREND_STATE = BASE_DIR / "supertrend" / "paper_trades" / "supertrend_state.json"


def load_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            df = pd.read_csv(path)
            if not df.empty:
                return df
        except Exception:
            pass
    return pd.DataFrame()


def load_state(path: Path) -> dict:
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def normalize_ema_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize EMA trade log to common format."""
    if df.empty:
        return pd.DataFrame()
    rows = []
    for _, r in df.iterrows():
        entry_time = str(r.get("entry_time", ""))[:16]
        exit_time = str(r.get("exit_time", ""))[:16]
        date_str = entry_time[:10] if entry_time else ""
        rows.append({
            "date": date_str,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "strategy": "EMA Crossover",
            "direction": r.get("direction", "N/A"),
            "instrument": f"{r.get('option_type','')}{r.get('strike','')}",
            "entry_price": r.get("entry_premium", r.get("entry_spot", "")),
            "exit_price": r.get("exit_premium", r.get("exit_spot", "")),
            "qty": r.get("quantity", 25),
            "gross_pnl": round(float(r.get("gross_pnl", 0)), 2),
            "costs": round(float(r.get("costs", 0)), 2),
            "net_pnl": round(float(r.get("net_pnl", 0)), 2),
            "status": r.get("status", "N/A"),
        })
    return pd.DataFrame(rows)


def normalize_sapphire_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Sapphire trade log to common format."""
    if df.empty:
        return pd.DataFrame()
    rows = []
    for _, r in df.iterrows():
        date_str = str(r.get("date", ""))[:10]
        rows.append({
            "date": date_str,
            "entry_time": date_str,
            "exit_time": date_str,
            "strategy": "Sapphire Strangle",
            "direction": "STRANGLE",
            "instrument": f"CE{r.get('ce_strike','')} / PE{r.get('pe_strike','')}",
            "entry_price": round(float(r.get("ce_entry_prem", 0)) + float(r.get("pe_entry_prem", 0)), 2),
            "exit_price": round(float(r.get("ce_exit_prem", 0)) + float(r.get("pe_exit_prem", 0)), 2),
            "qty": 25,
            "gross_pnl": round(float(r.get("gross_pnl", 0)), 2),
            "costs": round(float(r.get("costs", 0)), 2),
            "net_pnl": round(float(r.get("net_pnl", 0)), 2),
            "status": r.get("exit_reason", "N/A"),
        })
    return pd.DataFrame(rows)


def normalize_momentum_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Momentum trade log to common format."""
    if df.empty:
        return pd.DataFrame()
    rows = []
    for _, r in df.iterrows():
        date_str = str(r.get("date", ""))[:10]
        rows.append({
            "date": date_str,
            "entry_time": str(r.get("entry_time", date_str))[:16],
            "exit_time": str(r.get("exit_time", date_str))[:16],
            "strategy": "Momentum",
            "direction": r.get("direction", "N/A"),
            "instrument": "NIFTY FUT",
            "entry_price": r.get("entry_price", ""),
            "exit_price": r.get("exit_price", ""),
            "qty": r.get("quantity", 25),
            "gross_pnl": round(float(r.get("gross_pnl", 0)), 2),
            "costs": round(float(r.get("costs", 0)), 2),
            "net_pnl": round(float(r.get("net_pnl", 0)), 2),
            "status": r.get("status", "N/A"),
        })
    return pd.DataFrame(rows)


def normalize_supertrend_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Supertrend trade log to common format."""
    if df.empty:
        return pd.DataFrame()
    rows = []
    for _, r in df.iterrows():
        date_str = str(r.get("date", ""))[:10]
        rows.append({
            "date": date_str,
            "entry_time": str(r.get("entry_time", date_str))[:16],
            "exit_time": str(r.get("exit_time", date_str))[:16],
            "strategy": "Supertrend VWAP",
            "direction": r.get("direction", "N/A"),
            "instrument": "NIFTY FUT",
            "entry_price": r.get("entry_price", ""),
            "exit_price": r.get("exit_price", ""),
            "qty": r.get("quantity", 25),
            "gross_pnl": round(float(r.get("gross_pnl", 0)), 2),
            "costs": round(float(r.get("costs", 0)), 2),
            "net_pnl": round(float(r.get("net_pnl", 0)), 2),
            "status": r.get("status", "N/A"),
        })
    return pd.DataFrame(rows)


def calc_strategy_stats(df: pd.DataFrame) -> dict:
    """Compute stats for a strategy's trades."""
    if df.empty:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
                "gross_pnl": 0, "costs": 0, "net_pnl": 0,
                "avg_win": 0, "avg_loss": 0, "best": 0, "worst": 0,
                "profit_factor": 0, "max_streak_w": 0, "max_streak_l": 0}

    pnl = df["net_pnl"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    total_wins = wins.sum() if len(wins) > 0 else 0
    total_losses = abs(losses.sum()) if len(losses) > 0 else 0

    # Streaks
    max_w = max_l = cur_w = cur_l = 0
    for p in pnl:
        if p > 0:
            cur_w += 1; cur_l = 0
            max_w = max(max_w, cur_w)
        else:
            cur_l += 1; cur_w = 0
            max_l = max(max_l, cur_l)

    return {
        "trades": len(pnl),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(pnl) * 100, 1) if len(pnl) > 0 else 0,
        "gross_pnl": round(df["gross_pnl"].sum(), 2),
        "costs": round(df["costs"].sum(), 2),
        "net_pnl": round(pnl.sum(), 2),
        "avg_win": round(wins.mean(), 2) if len(wins) > 0 else 0,
        "avg_loss": round(losses.mean(), 2) if len(losses) > 0 else 0,
        "best": round(pnl.max(), 2) if len(pnl) > 0 else 0,
        "worst": round(pnl.min(), 2) if len(pnl) > 0 else 0,
        "profit_factor": round(total_wins / total_losses, 2) if total_losses > 0 else float('inf') if total_wins > 0 else 0,
        "max_streak_w": max_w,
        "max_streak_l": max_l,
    }


def generate_tradebook() -> str:
    """Generate the complete trade book HTML."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Load and normalize all trades
    ema_df = normalize_ema_trades(load_csv(EMA_TRADE_LOG))
    sap_df = normalize_sapphire_trades(load_csv(SAPPHIRE_TRADE_LOG))
    mom_df = normalize_momentum_trades(load_csv(MOMENTUM_TRADE_LOG))
    st_df = normalize_supertrend_trades(load_csv(SUPERTREND_TRADE_LOG))

    all_trades = pd.concat([ema_df, sap_df, mom_df, st_df], ignore_index=True)
    if not all_trades.empty:
        all_trades = all_trades.sort_values("entry_time", ascending=False).reset_index(drop=True)

    # Stats per strategy
    stats = {
        "EMA Crossover": calc_strategy_stats(ema_df),
        "Sapphire Strangle": calc_strategy_stats(sap_df),
        "Momentum": calc_strategy_stats(mom_df),
        "Supertrend VWAP": calc_strategy_stats(st_df),
    }
    combined = calc_strategy_stats(all_trades)

    # Strategy colors
    colors = {
        "EMA Crossover": "#22c55e",
        "Sapphire Strangle": "#3b82f6",
        "Momentum": "#f97316",
        "Supertrend VWAP": "#14b8a6",
    }

    # Build trade table rows as JSON for JavaScript sorting/filtering
    trades_json = "[]"
    if not all_trades.empty:
        records = []
        for _, r in all_trades.iterrows():
            records.append({
                "date": str(r.get("date", "")),
                "entry_time": str(r.get("entry_time", "")),
                "exit_time": str(r.get("exit_time", "")),
                "strategy": str(r.get("strategy", "")),
                "direction": str(r.get("direction", "")),
                "instrument": str(r.get("instrument", "")),
                "entry_price": float(r["entry_price"]) if r.get("entry_price") not in ["", None, "nan"] else 0,
                "exit_price": float(r["exit_price"]) if r.get("exit_price") not in ["", None, "nan"] else 0,
                "qty": int(r.get("qty", 25)),
                "gross_pnl": float(r.get("gross_pnl", 0)),
                "costs": float(r.get("costs", 0)),
                "net_pnl": float(r.get("net_pnl", 0)),
                "status": str(r.get("status", "")),
            })
        import json as _json
        trades_json = _json.dumps(records)

    # Stats JSON
    import json as _json
    stats_json = _json.dumps(stats)

    def pnl_color(val):
        if val > 0: return "#00c853"
        elif val < 0: return "#ff1744"
        return "#8b949e"

    def fmt(val):
        return f"₹{val:+,.2f}" if val != 0 else "₹0.00"

    # Unique dates for date picker
    date_min = ""
    date_max = ""
    if not all_trades.empty:
        dates = all_trades["date"].dropna().unique()
        if len(dates) > 0:
            date_min = sorted(dates)[0]
            date_max = sorted(dates)[-1]

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trade Book - Nifty Algo</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0d1117;color:#e6edf3;padding:20px;max-width:1400px;margin:0 auto}}
a{{color:#58a6ff;text-decoration:none}}
a:hover{{text-decoration:underline}}

.header{{text-align:center;padding:20px 0;border-bottom:1px solid #30363d;margin-bottom:24px}}
.header h1{{color:#58a6ff;font-size:24px}}
.header .subtitle{{color:#8b949e;font-size:14px;margin-top:4px}}
.header .nav{{margin-top:8px;font-size:13px}}

/* Summary Cards */
.summary-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:24px}}
.s-card{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;text-align:center}}
.s-card .s-label{{color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:0.5px}}
.s-card .s-val{{font-size:24px;font-weight:700;margin-top:4px}}
.s-card .s-sub{{color:#8b949e;font-size:12px;margin-top:2px}}

/* Strategy Tabs */
.tabs{{display:flex;gap:8px;margin-bottom:20px;flex-wrap:wrap;align-items:center}}
.tab{{padding:8px 16px;border-radius:8px;cursor:pointer;font-size:13px;font-weight:600;border:1px solid #30363d;background:#161b22;color:#8b949e;transition:all 0.2s}}
.tab:hover{{border-color:#58a6ff;color:#e6edf3}}
.tab.active{{background:#1f6feb22;border-color:#1f6feb;color:#58a6ff}}
.tab .dot{{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}}
.tab .count{{background:#30363d;padding:2px 8px;border-radius:10px;margin-left:6px;font-size:11px}}

/* Filters */
.filters{{display:flex;gap:12px;margin-bottom:16px;align-items:center;flex-wrap:wrap}}
.filters label{{color:#8b949e;font-size:12px}}
.filters input,.filters select{{background:#0d1117;border:1px solid #30363d;color:#e6edf3;padding:6px 10px;border-radius:6px;font-size:13px}}
.filters input:focus,.filters select:focus{{outline:none;border-color:#58a6ff}}
.filter-btn{{padding:6px 14px;border-radius:6px;cursor:pointer;font-size:12px;border:1px solid #30363d;background:#161b22;color:#8b949e}}
.filter-btn:hover{{border-color:#58a6ff;color:#e6edf3}}

/* Strategy Stats Panel */
.strat-stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;margin-bottom:24px}}
.strat-panel{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;border-top:3px solid #30363d}}
.strat-panel h3{{font-size:14px;margin-bottom:12px}}
.strat-panel .sg{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}}
.strat-panel .si{{text-align:center;padding:8px;background:#0d1117;border-radius:6px}}
.strat-panel .si .sl{{color:#8b949e;font-size:10px;text-transform:uppercase}}
.strat-panel .si .sv{{font-size:16px;font-weight:700;margin-top:2px}}

/* Trade Table */
.table-wrap{{background:#161b22;border:1px solid #30363d;border-radius:10px;overflow:hidden;margin-bottom:20px}}
.table-info{{display:flex;justify-content:space-between;align-items:center;padding:12px 16px;border-bottom:1px solid #30363d}}
.table-info .ti-left{{color:#8b949e;font-size:13px}}
.table-info .ti-right{{color:#8b949e;font-size:12px}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{text-align:left;color:#8b949e;padding:10px 12px;border-bottom:1px solid #30363d;font-weight:600;cursor:pointer;user-select:none;white-space:nowrap}}
th:hover{{color:#58a6ff}}
th .sort-icon{{margin-left:4px;font-size:10px;opacity:0.5}}
th.sorted .sort-icon{{opacity:1;color:#58a6ff}}
td{{padding:8px 12px;border-bottom:1px solid #21262d;white-space:nowrap}}
tr:hover{{background:#1c2128}}
tr.win{{border-left:3px solid #00c853}}
tr.loss{{border-left:3px solid #ff1744}}
.pnl-pos{{color:#00c853;font-weight:700}}
.pnl-neg{{color:#ff1744;font-weight:700}}
.pnl-zero{{color:#8b949e}}
.strat-badge{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;color:#fff}}
.dir-buy{{color:#00c853}}
.dir-sell{{color:#ff1744}}
.dir-strangle{{color:#a78bfa}}

.footer{{text-align:center;color:#484f58;font-size:12px;padding:20px 0}}

/* Pagination */
.pagination{{display:flex;justify-content:center;align-items:center;gap:8px;padding:16px}}
.page-btn{{padding:6px 12px;border-radius:6px;cursor:pointer;font-size:12px;border:1px solid #30363d;background:#161b22;color:#8b949e}}
.page-btn:hover{{border-color:#58a6ff;color:#e6edf3}}
.page-btn.active{{background:#1f6feb22;border-color:#1f6feb;color:#58a6ff}}
.page-btn:disabled{{opacity:0.3;cursor:default}}
.page-info{{color:#8b949e;font-size:12px}}

@media(max-width:768px){{
  body{{padding:12px}}
  .summary-cards{{grid-template-columns:repeat(2,1fr)}}
  .strat-stats{{grid-template-columns:1fr}}
  table{{font-size:11px}}
  td,th{{padding:6px 8px}}
}}
</style>
</head>
<body>

<div class="header">
  <h1>📒 Trade Book</h1>
  <div class="subtitle">All Paper Trading Activity | Updated {now_str}</div>
  <div class="nav"><a href="dashboard.html">← Dashboard</a></div>
</div>

<!-- Combined Summary -->
<div class="summary-cards">
  <div class="s-card">
    <div class="s-label">Total Trades</div>
    <div class="s-val">{combined["trades"]}</div>
    <div class="s-sub">{combined["wins"]}W / {combined["losses"]}L</div>
  </div>
  <div class="s-card">
    <div class="s-label">Net P&amp;L</div>
    <div class="s-val" style="color:{pnl_color(combined["net_pnl"])}">{fmt(combined["net_pnl"])}</div>
    <div class="s-sub">Gross: {fmt(combined["gross_pnl"])} | Costs: {fmt(-combined["costs"])}</div>
  </div>
  <div class="s-card">
    <div class="s-label">Win Rate</div>
    <div class="s-val">{combined["win_rate"]}%</div>
    <div class="s-sub">PF: {combined["profit_factor"]}</div>
  </div>
  <div class="s-card">
    <div class="s-label">Best / Worst</div>
    <div class="s-val" style="font-size:16px"><span style="color:#00c853">{fmt(combined["best"])}</span> / <span style="color:#ff1744">{fmt(combined["worst"])}</span></div>
    <div class="s-sub">Avg Win: {fmt(combined["avg_win"])} | Avg Loss: {fmt(combined["avg_loss"])}</div>
  </div>
  <div class="s-card">
    <div class="s-label">Streaks</div>
    <div class="s-val" style="font-size:16px"><span style="color:#00c853">{combined["max_streak_w"]}W</span> / <span style="color:#ff1744">{combined["max_streak_l"]}L</span></div>
    <div class="s-sub">Max consecutive</div>
  </div>
</div>

<!-- Strategy Stats -->
<div class="strat-stats">
''' + ''.join([f'''  <div class="strat-panel" style="border-top-color:{colors[name]}">
    <h3 style="color:{colors[name]}">{name}</h3>
    <div class="sg">
      <div class="si"><div class="sl">Trades</div><div class="sv">{s["trades"]}</div></div>
      <div class="si"><div class="sl">Win Rate</div><div class="sv">{s["win_rate"]}%</div></div>
      <div class="si"><div class="sl">Net P&amp;L</div><div class="sv" style="color:{pnl_color(s["net_pnl"])}">{fmt(s["net_pnl"])}</div></div>
      <div class="si"><div class="sl">Avg Win</div><div class="sv" style="color:#00c853">{fmt(s["avg_win"])}</div></div>
      <div class="si"><div class="sl">Avg Loss</div><div class="sv" style="color:#ff1744">{fmt(s["avg_loss"])}</div></div>
      <div class="si"><div class="sl">Profit Factor</div><div class="sv">{s["profit_factor"]}</div></div>
    </div>
  </div>
''' for name, s in stats.items()]) + f'''</div>

<!-- Tabs & Filters -->
<div class="tabs" id="stratTabs">
  <div class="tab active" data-strategy="ALL" onclick="filterStrategy('ALL',this)">
    <span class="dot" style="background:#58a6ff"></span>All<span class="count" id="count-ALL">{combined["trades"]}</span>
  </div>
  <div class="tab" data-strategy="EMA Crossover" onclick="filterStrategy('EMA Crossover',this)">
    <span class="dot" style="background:#22c55e"></span>EMA<span class="count" id="count-EMA">{stats["EMA Crossover"]["trades"]}</span>
  </div>
  <div class="tab" data-strategy="Sapphire Strangle" onclick="filterStrategy('Sapphire Strangle',this)">
    <span class="dot" style="background:#3b82f6"></span>Sapphire<span class="count" id="count-SAP">{stats["Sapphire Strangle"]["trades"]}</span>
  </div>
  <div class="tab" data-strategy="Momentum" onclick="filterStrategy('Momentum',this)">
    <span class="dot" style="background:#f97316"></span>Momentum<span class="count" id="count-MOM">{stats["Momentum"]["trades"]}</span>
  </div>
  <div class="tab" data-strategy="Supertrend VWAP" onclick="filterStrategy('Supertrend VWAP',this)">
    <span class="dot" style="background:#14b8a6"></span>Supertrend<span class="count" id="count-ST">{stats["Supertrend VWAP"]["trades"]}</span>
  </div>
</div>
<div class="filters">
  <label>From</label> <input type="date" id="dateFrom" value="{date_min}" onchange="applyFilters()">
  <label>To</label> <input type="date" id="dateTo" value="{date_max}" onchange="applyFilters()">
  <label>P&amp;L</label>
  <select id="pnlFilter" onchange="applyFilters()">
    <option value="ALL">All</option>
    <option value="WIN">Winners only</option>
    <option value="LOSS">Losers only</option>
  </select>
  <label>Direction</label>
  <select id="dirFilter" onchange="applyFilters()">
    <option value="ALL">All</option>
    <option value="BUY">Buy / Long</option>
    <option value="SELL">Sell / Short</option>
    <option value="STRANGLE">Strangle</option>
  </select>
  <button class="filter-btn" onclick="resetFilters()">Reset</button>
  <button class="filter-btn" onclick="exportCSV()">📥 Export CSV</button>
</div>

<!-- Trade Table -->
<div class="table-wrap">
  <div class="table-info">
    <div class="ti-left" id="tableInfo">Showing 0 trades</div>
    <div class="ti-right" id="filteredPnl"></div>
  </div>
  <div style="overflow-x:auto">
    <table id="tradeTable">
      <thead>
        <tr>
          <th onclick="sortTable('date')">#<span class="sort-icon">⇅</span></th>
          <th onclick="sortTable('date')">Date<span class="sort-icon">⇅</span></th>
          <th onclick="sortTable('strategy')">Strategy<span class="sort-icon">⇅</span></th>
          <th onclick="sortTable('direction')">Dir<span class="sort-icon">⇅</span></th>
          <th>Instrument</th>
          <th onclick="sortTable('entry_price')">Entry<span class="sort-icon">⇅</span></th>
          <th onclick="sortTable('exit_price')">Exit<span class="sort-icon">⇅</span></th>
          <th>Qty</th>
          <th onclick="sortTable('gross_pnl')">Gross<span class="sort-icon">⇅</span></th>
          <th onclick="sortTable('costs')">Costs<span class="sort-icon">⇅</span></th>
          <th onclick="sortTable('net_pnl')">Net P&amp;L<span class="sort-icon">⇅</span></th>
          <th onclick="sortTable('net_pnl')">Cum P&amp;L<span class="sort-icon">⇅</span></th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody id="tradeBody"></tbody>
    </table>
  </div>
  <div class="pagination" id="pagination"></div>
</div>

<div class="footer">
  Nifty Algo Trader — Trade Book — <a href="dashboard.html">Dashboard</a>
</div>

<script>
const ALL_TRADES = {trades_json};
const STRAT_COLORS = {_json.dumps(colors)};
const PAGE_SIZE = 50;

let filtered = [...ALL_TRADES];
let currentPage = 1;
let currentStrategy = 'ALL';
let sortCol = 'date';
let sortAsc = false;

function pnlClass(v) {{ return v > 0 ? 'pnl-pos' : v < 0 ? 'pnl-neg' : 'pnl-zero'; }}
function fmt(v) {{ return v === 0 ? '₹0' : (v > 0 ? '₹+' : '₹') + v.toFixed(2).replace(/\\B(?=(\\d{{3}})+(?!\\d))/g, ','); }}
function dirClass(d) {{
  d = d.toUpperCase();
  if (d === 'BUY' || d === 'LONG') return 'dir-buy';
  if (d === 'SELL' || d === 'SHORT') return 'dir-sell';
  if (d === 'STRANGLE') return 'dir-strangle';
  return '';
}}

function filterStrategy(strat, el) {{
  currentStrategy = strat;
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  if (el) el.classList.add('active');
  applyFilters();
}}

function applyFilters() {{
  const dateFrom = document.getElementById('dateFrom').value;
  const dateTo = document.getElementById('dateTo').value;
  const pnlF = document.getElementById('pnlFilter').value;
  const dirF = document.getElementById('dirFilter').value;

  filtered = ALL_TRADES.filter(t => {{
    if (currentStrategy !== 'ALL' && t.strategy !== currentStrategy) return false;
    if (dateFrom && t.date < dateFrom) return false;
    if (dateTo && t.date > dateTo) return false;
    if (pnlF === 'WIN' && t.net_pnl <= 0) return false;
    if (pnlF === 'LOSS' && t.net_pnl > 0) return false;
    if (dirF !== 'ALL') {{
      const d = t.direction.toUpperCase();
      if (dirF === 'BUY' && d !== 'BUY' && d !== 'LONG') return false;
      if (dirF === 'SELL' && d !== 'SELL' && d !== 'SHORT') return false;
      if (dirF === 'STRANGLE' && d !== 'STRANGLE') return false;
    }}
    return true;
  }});

  doSort();
  currentPage = 1;
  renderTable();
}}

function sortTable(col) {{
  if (sortCol === col) sortAsc = !sortAsc;
  else {{ sortCol = col; sortAsc = true; }}
  doSort();
  renderTable();
}}

function doSort() {{
  filtered.sort((a, b) => {{
    let va = a[sortCol], vb = b[sortCol];
    if (typeof va === 'number') return sortAsc ? va - vb : vb - va;
    va = String(va); vb = String(vb);
    return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
  }});
}}

function renderTable() {{
  const tbody = document.getElementById('tradeBody');
  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  if (currentPage > totalPages) currentPage = totalPages;
  const start = (currentPage - 1) * PAGE_SIZE;
  const page = filtered.slice(start, start + PAGE_SIZE);

  // Cumulative P&L (across ALL filtered, not just page)
  let cumPnl = 0;
  const cumArr = [];
  // Sort by date asc for cumulative calc
  const sortedAll = [...filtered].sort((a,b) => a.date < b.date ? -1 : a.date > b.date ? 1 : 0);
  const cumMap = new Map();
  sortedAll.forEach((t, i) => {{
    cumPnl += t.net_pnl;
    // Use index in filtered array as key
    const key = t.entry_time + t.strategy + t.net_pnl;
    cumMap.set(key, cumPnl);
  }});

  let html = '';
  page.forEach((t, i) => {{
    const idx = start + i + 1;
    const rowClass = t.net_pnl > 0 ? 'win' : t.net_pnl < 0 ? 'loss' : '';
    const color = STRAT_COLORS[t.strategy] || '#8b949e';
    const key = t.entry_time + t.strategy + t.net_pnl;
    const cum = cumMap.get(key) || 0;
    html += `<tr class="${{rowClass}}">
      <td style="color:#8b949e">${{idx}}</td>
      <td>${{t.date}}</td>
      <td><span class="strat-badge" style="background:${{color}}33;color:${{color}}">${{t.strategy}}</span></td>
      <td class="${{dirClass(t.direction)}}">${{t.direction}}</td>
      <td>${{t.instrument}}</td>
      <td>${{t.entry_price ? t.entry_price.toFixed(2) : '-'}}</td>
      <td>${{t.exit_price ? t.exit_price.toFixed(2) : '-'}}</td>
      <td>${{t.qty}}</td>
      <td class="${{pnlClass(t.gross_pnl)}}">${{fmt(t.gross_pnl)}}</td>
      <td style="color:#f59e0b">${{fmt(-t.costs)}}</td>
      <td class="${{pnlClass(t.net_pnl)}}">${{fmt(t.net_pnl)}}</td>
      <td class="${{pnlClass(cum)}}">${{fmt(cum)}}</td>
      <td style="color:#8b949e;font-size:11px">${{t.status}}</td>
    </tr>`;
  }});
  tbody.innerHTML = html || '<tr><td colspan="13" style="text-align:center;padding:40px;color:#8b949e">No trades found</td></tr>';

  // Info bar
  const totalPnl = filtered.reduce((s, t) => s + t.net_pnl, 0);
  const wins = filtered.filter(t => t.net_pnl > 0).length;
  document.getElementById('tableInfo').textContent = `Showing ${{filtered.length}} trades (${{wins}} wins, ${{filtered.length - wins}} losses)`;
  document.getElementById('filteredPnl').innerHTML = `Filtered P&L: <span style="color:${{totalPnl >= 0 ? '#00c853' : '#ff1744'}};font-weight:bold">${{fmt(totalPnl)}}</span>`;

  // Pagination
  let pgHtml = `<button class="page-btn" onclick="goPage(1)" ${{currentPage===1?'disabled':''}}>«</button>`;
  pgHtml += `<button class="page-btn" onclick="goPage(${{currentPage-1}})" ${{currentPage===1?'disabled':''}}>‹</button>`;
  const maxBtns = 7;
  let pStart = Math.max(1, currentPage - 3);
  let pEnd = Math.min(totalPages, pStart + maxBtns - 1);
  if (pEnd - pStart < maxBtns - 1) pStart = Math.max(1, pEnd - maxBtns + 1);
  for (let p = pStart; p <= pEnd; p++) {{
    pgHtml += `<button class="page-btn ${{p===currentPage?'active':''}}" onclick="goPage(${{p}})">${{p}}</button>`;
  }}
  pgHtml += `<button class="page-btn" onclick="goPage(${{currentPage+1}})" ${{currentPage>=totalPages?'disabled':''}}>›</button>`;
  pgHtml += `<button class="page-btn" onclick="goPage(${{totalPages}})" ${{currentPage>=totalPages?'disabled':''}}>»</button>`;
  pgHtml += `<span class="page-info">${{currentPage}} / ${{totalPages}}</span>`;
  document.getElementById('pagination').innerHTML = pgHtml;
}}

function goPage(p) {{
  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  currentPage = Math.max(1, Math.min(p, totalPages));
  renderTable();
  document.getElementById('tradeTable').scrollIntoView({{behavior:'smooth'}});
}}

function resetFilters() {{
  document.getElementById('dateFrom').value = '{date_min}';
  document.getElementById('dateTo').value = '{date_max}';
  document.getElementById('pnlFilter').value = 'ALL';
  document.getElementById('dirFilter').value = 'ALL';
  filterStrategy('ALL', document.querySelector('.tab[data-strategy="ALL"]'));
}}

function exportCSV() {{
  if (filtered.length === 0) return;
  const headers = ['Date','Entry Time','Exit Time','Strategy','Direction','Instrument','Entry Price','Exit Price','Qty','Gross PnL','Costs','Net PnL','Status'];
  let csv = headers.join(',') + '\\n';
  filtered.forEach(t => {{
    csv += [t.date,t.entry_time,t.exit_time,t.strategy,t.direction,`"${{t.instrument}}"`,t.entry_price,t.exit_price,t.qty,t.gross_pnl,t.costs,t.net_pnl,t.status].join(',') + '\\n';
  }});
  const blob = new Blob([csv], {{type:'text/csv'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'tradebook_' + new Date().toISOString().slice(0,10) + '.csv';
  a.click();
}}

// Initial render
applyFilters();
</script>
</body>
</html>'''
    return html


def main():
    parser = argparse.ArgumentParser(description="Trade Book Generator")
    parser.add_argument("--open", action="store_true", help="Open in browser after generating")
    args = parser.parse_args()

    print("\n  Generating Trade Book...")

    html = generate_tradebook()
    with open(TRADEBOOK_FILE, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  Saved: {TRADEBOOK_FILE}")

    # Quick summary
    ema_count = len(load_csv(EMA_TRADE_LOG))
    sap_count = len(load_csv(SAPPHIRE_TRADE_LOG))
    mom_count = len(load_csv(MOMENTUM_TRADE_LOG))
    st_count = len(load_csv(SUPERTREND_TRADE_LOG))
    total = ema_count + sap_count + mom_count + st_count
    print(f"  Trades: EMA={ema_count}, Sapphire={sap_count}, Momentum={mom_count}, Supertrend={st_count} | Total={total}")

    if args.open:
        import webbrowser
        webbrowser.open(str(TRADEBOOK_FILE))

    print("  Done.\n")


if __name__ == "__main__":
    main()
