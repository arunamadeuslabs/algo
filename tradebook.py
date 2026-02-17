"""
Trade Book Generator
=====================
Generates a comprehensive HTML trade book page showing ALL trades
across all strategies (EMA, Sapphire, Momentum, Supertrend).

Features:
  - Daily / Weekly / Monthly aggregation views
  - 1M / 3M / 6M / All-Time period quick-select
  - Strategy filter dropdown
  - Expandable grouped views with individual trade details
  - Export CSV
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
    """Generate the complete trade book HTML with D/W/M views and period tabs."""
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

    colors = {
        "EMA Crossover": "#22c55e",
        "Sapphire Strangle": "#3b82f6",
        "Momentum": "#f97316",
        "Supertrend VWAP": "#14b8a6",
    }

    # Build trades JSON
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
        trades_json = json.dumps(records)

    colors_json = json.dumps(colors)

    def pnl_color(val):
        if val > 0: return "#00c853"
        elif val < 0: return "#ff1744"
        return "#8b949e"

    def fmt(val):
        return f"\u20b9{val:+,.2f}" if val != 0 else "\u20b90.00"

    # Strategy stats panels
    strat_panels = ""
    for name, s in stats.items():
        c = colors[name]
        strat_panels += f'''  <div class="strat-panel" style="border-top-color:{c}">
    <h3 style="color:{c}">{name}</h3>
    <div class="sg">
      <div class="si"><div class="sl">Trades</div><div class="sv">{s["trades"]}</div></div>
      <div class="si"><div class="sl">Win Rate</div><div class="sv">{s["win_rate"]}%</div></div>
      <div class="si"><div class="sl">Net P&amp;L</div><div class="sv" style="color:{pnl_color(s["net_pnl"])}">{fmt(s["net_pnl"])}</div></div>
      <div class="si"><div class="sl">Avg Win</div><div class="sv" style="color:#00c853">{fmt(s["avg_win"])}</div></div>
      <div class="si"><div class="sl">Avg Loss</div><div class="sv" style="color:#ff1744">{fmt(s["avg_loss"])}</div></div>
      <div class="si"><div class="sl">PF</div><div class="sv">{s["profit_factor"]}</div></div>
    </div>
  </div>
'''

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
.summary-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:24px}}
.s-card{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;text-align:center}}
.s-card .s-label{{color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:0.5px}}
.s-card .s-val{{font-size:24px;font-weight:700;margin-top:4px}}
.s-card .s-sub{{color:#8b949e;font-size:12px;margin-top:2px}}
.strat-stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;margin-bottom:24px}}
.strat-panel{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;border-top:3px solid #30363d}}
.strat-panel h3{{font-size:14px;margin-bottom:12px}}
.strat-panel .sg{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}}
.strat-panel .si{{text-align:center;padding:8px;background:#0d1117;border-radius:6px}}
.strat-panel .si .sl{{color:#8b949e;font-size:10px;text-transform:uppercase}}
.strat-panel .si .sv{{font-size:16px;font-weight:700;margin-top:2px}}
.tabs-row{{display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap;align-items:center}}
.tab-label{{color:#8b949e;font-size:12px;font-weight:600;margin-right:4px;text-transform:uppercase;letter-spacing:0.5px}}
.tab{{padding:8px 16px;border-radius:8px;cursor:pointer;font-size:13px;font-weight:600;border:1px solid #30363d;background:#161b22;color:#8b949e;transition:all 0.2s}}
.tab:hover{{border-color:#58a6ff;color:#e6edf3}}
.tab.active{{background:#1f6feb22;border-color:#1f6feb;color:#58a6ff}}
.filters{{display:flex;gap:12px;margin-bottom:16px;align-items:center;flex-wrap:wrap}}
.filters label{{color:#8b949e;font-size:12px}}
.filters select{{background:#0d1117;border:1px solid #30363d;color:#e6edf3;padding:6px 10px;border-radius:6px;font-size:13px}}
.filters select:focus{{outline:none;border-color:#58a6ff}}
.filter-btn{{padding:6px 14px;border-radius:6px;cursor:pointer;font-size:12px;border:1px solid #30363d;background:#161b22;color:#8b949e}}
.filter-btn:hover{{border-color:#58a6ff;color:#e6edf3}}
.table-wrap{{background:#161b22;border:1px solid #30363d;border-radius:10px;overflow:hidden;margin-bottom:20px}}
.table-info{{display:flex;justify-content:space-between;align-items:center;padding:12px 16px;border-bottom:1px solid #30363d}}
.table-info .ti-left{{color:#8b949e;font-size:13px}}
.table-info .ti-right{{color:#8b949e;font-size:12px}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{text-align:left;color:#8b949e;padding:10px 12px;border-bottom:1px solid #30363d;font-weight:600;white-space:nowrap}}
td{{padding:8px 12px;border-bottom:1px solid #21262d;white-space:nowrap}}
tr:hover{{background:#1c2128}}
.pnl-pos{{color:#00c853;font-weight:700}}
.pnl-neg{{color:#ff1744;font-weight:700}}
.pnl-zero{{color:#8b949e}}
.strat-badge{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;color:#fff}}
.dir-buy{{color:#00c853}}
.dir-sell{{color:#ff1744}}
.dir-strangle{{color:#a78bfa}}
.group-row{{cursor:pointer;background:#161b22}}
.group-row:hover{{background:#1c2128}}
.group-row td{{padding:10px 12px;font-weight:600}}
.expand-icon{{display:inline-block;width:16px;color:#58a6ff;font-size:10px;transition:transform 0.2s}}
.detail-table{{width:100%;border-collapse:collapse;font-size:11px;background:#0d1117;margin:4px 0}}
.detail-table th{{color:#8b949e;padding:6px 10px;font-size:10px;text-transform:uppercase;border-bottom:1px solid #21262d}}
.detail-table td{{padding:6px 10px;border-bottom:1px solid #161b22}}
.detail-table tr:hover{{background:#161b2288}}
.footer{{text-align:center;color:#484f58;font-size:12px;padding:20px 0}}
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
  <h1>Trade Book</h1>
  <div class="subtitle">All Paper Trading Activity | Updated {now_str}</div>
  <div class="nav"><a href="dashboard.html">&larr; Dashboard</a></div>
</div>

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

<div class="strat-stats">
{strat_panels}</div>

<!-- Period Tabs -->
<div class="tabs-row" id="periodTabs">
  <span class="tab-label">Period:</span>
  <div class="tab" data-period="1M" onclick="setPeriod('1M',this)">1 Month</div>
  <div class="tab" data-period="3M" onclick="setPeriod('3M',this)">3 Months</div>
  <div class="tab" data-period="6M" onclick="setPeriod('6M',this)">6 Months</div>
  <div class="tab active" data-period="ALL" onclick="setPeriod('ALL',this)">All Time</div>
</div>

<!-- View Tabs -->
<div class="tabs-row" id="viewTabs">
  <span class="tab-label">View:</span>
  <div class="tab active" data-view="daily" onclick="setView('daily',this)">Daily</div>
  <div class="tab" data-view="weekly" onclick="setView('weekly',this)">Weekly</div>
  <div class="tab" data-view="monthly" onclick="setView('monthly',this)">Monthly</div>
</div>

<!-- Filters -->
<div class="filters">
  <label>Strategy</label>
  <select id="stratFilter" onchange="applyFilters()">
    <option value="ALL">All Strategies</option>
    <option value="EMA Crossover">EMA Crossover</option>
    <option value="Sapphire Strangle">Sapphire Strangle</option>
    <option value="Momentum">Momentum</option>
    <option value="Supertrend VWAP">Supertrend VWAP</option>
  </select>
  <button class="filter-btn" onclick="resetFilters()">Reset</button>
  <button class="filter-btn" onclick="exportCSV()">Export CSV</button>
</div>

<!-- Grouped Table -->
<div class="table-wrap">
  <div class="table-info">
    <div class="ti-left" id="tableInfo">Loading...</div>
    <div class="ti-right" id="filteredPnl"></div>
  </div>
  <div style="overflow-x:auto">
    <table id="groupTable">
      <thead>
        <tr>
          <th>Period</th>
          <th>Trades</th>
          <th style="color:#00c853">Wins</th>
          <th style="color:#ff1744">Losses</th>
          <th>Win%</th>
          <th>Gross P&amp;L</th>
          <th>Costs</th>
          <th>Net P&amp;L</th>
          <th>Cum P&amp;L</th>
        </tr>
      </thead>
      <tbody id="groupBody"></tbody>
    </table>
  </div>
</div>

<div class="footer">
  Nifty Algo Trader &mdash; Trade Book &mdash; <a href="dashboard.html">Dashboard</a>
</div>

<script>
const ALL_TRADES = {trades_json};
const STRAT_COLORS = {colors_json};

let periodFilter = 'ALL';
let viewMode = 'daily';
let strategyFilter = 'ALL';
let filtered = [];

function pnlClass(v) {{ return v > 0 ? 'pnl-pos' : v < 0 ? 'pnl-neg' : 'pnl-zero'; }}
function fmt(v) {{ return v === 0 ? '\u20b90' : (v > 0 ? '\u20b9+' : '\u20b9') + v.toFixed(2).replace(/\\B(?=(\\d{{3}})+(?!\\d))/g, ','); }}
function dirClass(d) {{
  d = (d||'').toUpperCase();
  if (d === 'BUY' || d === 'LONG') return 'dir-buy';
  if (d === 'SELL' || d === 'SHORT') return 'dir-sell';
  if (d === 'STRANGLE') return 'dir-strangle';
  return '';
}}

function getGroupKey(dateStr) {{
  if (viewMode === 'daily') return dateStr;
  if (viewMode === 'weekly') {{
    var d = new Date(dateStr + 'T00:00:00');
    var day = d.getDay();
    var diff = d.getDate() - day + (day === 0 ? -6 : 1);
    var monday = new Date(d);
    monday.setDate(diff);
    var mm = String(monday.getMonth() + 1).padStart(2, '0');
    var dd = String(monday.getDate()).padStart(2, '0');
    return monday.getFullYear() + '-' + mm + '-' + dd;
  }}
  if (viewMode === 'monthly') return dateStr.substring(0, 7);
  return dateStr;
}}

function getGroupLabel(key) {{
  var months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  if (viewMode === 'daily') {{
    var d = new Date(key + 'T00:00:00');
    var dayNames = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
    return dayNames[d.getDay()] + ' ' + d.getDate() + ' ' + months[d.getMonth()] + ' ' + d.getFullYear();
  }}
  if (viewMode === 'weekly') {{
    var d = new Date(key + 'T00:00:00');
    var end = new Date(d);
    end.setDate(end.getDate() + 4);
    return d.getDate() + ' ' + months[d.getMonth()] + ' - ' + end.getDate() + ' ' + months[end.getMonth()] + ' ' + end.getFullYear();
  }}
  if (viewMode === 'monthly') {{
    var parts = key.split('-');
    return months[parseInt(parts[1])-1] + ' ' + parts[0];
  }}
  return key;
}}

function setPeriod(period, el) {{
  periodFilter = period;
  document.querySelectorAll('#periodTabs .tab').forEach(function(t) {{ t.classList.remove('active'); }});
  if (el) el.classList.add('active');
  applyFilters();
}}

function setView(mode, el) {{
  viewMode = mode;
  document.querySelectorAll('#viewTabs .tab').forEach(function(t) {{ t.classList.remove('active'); }});
  if (el) el.classList.add('active');
  renderGrouped();
}}

function applyFilters() {{
  strategyFilter = document.getElementById('stratFilter').value;
  var now = new Date();
  var cutoff = null;
  if (periodFilter === '1M') cutoff = new Date(now.getFullYear(), now.getMonth() - 1, now.getDate());
  else if (periodFilter === '3M') cutoff = new Date(now.getFullYear(), now.getMonth() - 3, now.getDate());
  else if (periodFilter === '6M') cutoff = new Date(now.getFullYear(), now.getMonth() - 6, now.getDate());

  filtered = ALL_TRADES.filter(function(t) {{
    if (strategyFilter !== 'ALL' && t.strategy !== strategyFilter) return false;
    if (cutoff && new Date(t.date + 'T00:00:00') < cutoff) return false;
    return true;
  }});
  renderGrouped();
}}

function renderGrouped() {{
  var groups = {{}};
  var groupOrder = [];
  filtered.forEach(function(t) {{
    var key = getGroupKey(t.date);
    if (!groups[key]) {{ groups[key] = []; groupOrder.push(key); }}
    groups[key].push(t);
  }});
  groupOrder.sort(function(a, b) {{ return b.localeCompare(a); }});

  // Cumulative P&L (chronological)
  var cumPnl = 0;
  var cumMap = {{}};
  groupOrder.slice().reverse().forEach(function(key) {{
    cumPnl += groups[key].reduce(function(s, t) {{ return s + t.net_pnl; }}, 0);
    cumMap[key] = cumPnl;
  }});

  var html = '';
  groupOrder.forEach(function(key, gi) {{
    var trades = groups[key];
    var label = getGroupLabel(key);
    var netPnl = trades.reduce(function(s, t) {{ return s + t.net_pnl; }}, 0);
    var grossPnl = trades.reduce(function(s, t) {{ return s + t.gross_pnl; }}, 0);
    var costs = trades.reduce(function(s, t) {{ return s + t.costs; }}, 0);
    var wins = trades.filter(function(t) {{ return t.net_pnl > 0; }}).length;
    var losses = trades.length - wins;
    var winRate = trades.length > 0 ? (wins / trades.length * 100).toFixed(1) : '0.0';
    var cum = cumMap[key] || 0;

    html += '<tr class="group-row" onclick="toggleGroup(' + gi + ')">';
    html += '<td><span class="expand-icon" id="icon-' + gi + '">\u25B6</span> ' + label + '</td>';
    html += '<td>' + trades.length + '</td>';
    html += '<td style="color:#00c853">' + wins + '</td>';
    html += '<td style="color:#ff1744">' + losses + '</td>';
    html += '<td>' + winRate + '%</td>';
    html += '<td class="' + pnlClass(grossPnl) + '">' + fmt(grossPnl) + '</td>';
    html += '<td style="color:#f59e0b">' + fmt(-costs) + '</td>';
    html += '<td class="' + pnlClass(netPnl) + '">' + fmt(netPnl) + '</td>';
    html += '<td class="' + pnlClass(cum) + '">' + fmt(cum) + '</td>';
    html += '</tr>';

    // Detail rows
    html += '<tr class="detail-container" id="detail-' + gi + '" style="display:none"><td colspan="9" style="padding:0 8px 8px 8px">';
    html += '<table class="detail-table"><thead><tr><th>#</th><th>Time</th><th>Strategy</th><th>Dir</th><th>Instrument</th><th>Entry</th><th>Exit</th><th>Qty</th><th>Gross</th><th>Costs</th><th>Net P&L</th><th>Status</th></tr></thead><tbody>';
    trades.forEach(function(t, i) {{
      var color = STRAT_COLORS[t.strategy] || '#8b949e';
      html += '<tr>';
      html += '<td style="color:#8b949e">' + (i+1) + '</td>';
      html += '<td>' + (t.entry_time || t.date) + '</td>';
      html += '<td><span class="strat-badge" style="background:' + color + '33;color:' + color + '">' + t.strategy + '</span></td>';
      html += '<td class="' + dirClass(t.direction) + '">' + t.direction + '</td>';
      html += '<td>' + t.instrument + '</td>';
      html += '<td>' + (t.entry_price ? t.entry_price.toFixed(2) : '-') + '</td>';
      html += '<td>' + (t.exit_price ? t.exit_price.toFixed(2) : '-') + '</td>';
      html += '<td>' + t.qty + '</td>';
      html += '<td class="' + pnlClass(t.gross_pnl) + '">' + fmt(t.gross_pnl) + '</td>';
      html += '<td style="color:#f59e0b">' + fmt(-t.costs) + '</td>';
      html += '<td class="' + pnlClass(t.net_pnl) + '">' + fmt(t.net_pnl) + '</td>';
      html += '<td style="color:#8b949e;font-size:11px">' + t.status + '</td>';
      html += '</tr>';
    }});
    html += '</tbody></table></td></tr>';
  }});

  document.getElementById('groupBody').innerHTML = html || '<tr><td colspan="9" style="text-align:center;padding:40px;color:#8b949e">No trades found</td></tr>';

  var totalPnl = filtered.reduce(function(s, t) {{ return s + t.net_pnl; }}, 0);
  var totalWins = filtered.filter(function(t) {{ return t.net_pnl > 0; }}).length;
  var viewLabel = viewMode === 'daily' ? 'days' : viewMode === 'weekly' ? 'weeks' : 'months';
  document.getElementById('tableInfo').textContent = groupOrder.length + ' ' + viewLabel + ' | ' + filtered.length + ' trades (' + totalWins + ' wins, ' + (filtered.length - totalWins) + ' losses)';
  document.getElementById('filteredPnl').innerHTML = 'Net P&L: <span style="color:' + (totalPnl >= 0 ? '#00c853' : '#ff1744') + ';font-weight:bold">' + fmt(totalPnl) + '</span>';
}}

function toggleGroup(idx) {{
  var detail = document.getElementById('detail-' + idx);
  var icon = document.getElementById('icon-' + idx);
  if (detail.style.display === 'none') {{
    detail.style.display = 'table-row';
    icon.textContent = '\u25BC';
  }} else {{
    detail.style.display = 'none';
    icon.textContent = '\u25B6';
  }}
}}

function resetFilters() {{
  document.getElementById('stratFilter').value = 'ALL';
  setPeriod('ALL', document.querySelector('#periodTabs .tab[data-period="ALL"]'));
  setView('daily', document.querySelector('#viewTabs .tab[data-view="daily"]'));
}}

function exportCSV() {{
  if (filtered.length === 0) return;
  var headers = ['Date','Entry Time','Exit Time','Strategy','Direction','Instrument','Entry Price','Exit Price','Qty','Gross PnL','Costs','Net PnL','Status'];
  var csv = headers.join(',') + '\\n';
  filtered.forEach(function(t) {{
    csv += [t.date,t.entry_time,t.exit_time,t.strategy,t.direction,'"'+t.instrument+'"',t.entry_price,t.exit_price,t.qty,t.gross_pnl,t.costs,t.net_pnl,t.status].join(',') + '\\n';
  }});
  var blob = new Blob([csv], {{type:'text/csv'}});
  var a = document.createElement('a');
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
