"""
Live Paper Trading Dashboard Server
=====================================
Serves a real-time dashboard showing all paper trades across all 4 strategies.
Reads trade logs and state files from each strategy's paper_trades/ directory.

Usage:
  python dashboard_server.py                # Start on http://localhost:8050
  python dashboard_server.py --port 8080    # Custom port
"""

import os
import sys
import json
import argparse
import webbrowser
from datetime import datetime, date
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse

# Fix encoding for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import pandas as pd
except ImportError:
    print("  pip install pandas")
    sys.exit(1)

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# ── Strategy Definitions ─────────────────────────────────────
STRATEGIES = {
    "ema": {
        "name": "EMA Crossover",
        "color": "#22c55e",
        "trade_csv": BASE_DIR / "backtest" / "paper_trades" / "paper_trade_log.csv",
        "daily_csv": BASE_DIR / "backtest" / "paper_trades" / "daily_summary.csv",
        "state_file": BASE_DIR / "backtest" / "paper_trades" / "state.json",
        "log_file": BASE_DIR / "backtest" / "paper_trades" / "paper_trading.log",
        "pnl_col": "net_pnl",
        "capital_col": "capital",
        "date_col": "entry_time",
    },
    "sapphire": {
        "name": "Sapphire Strangle",
        "color": "#3b82f6",
        "trade_csv": BASE_DIR / "sapphire" / "paper_trades" / "sapphire_trade_log.csv",
        "daily_csv": BASE_DIR / "sapphire" / "paper_trades" / "sapphire_daily_summary.csv",
        "state_file": BASE_DIR / "sapphire" / "paper_trades" / "sapphire_state.json",
        "log_file": BASE_DIR / "sapphire" / "paper_trades" / "sapphire_paper.log",
        "pnl_col": "net_pnl",
        "capital_col": "capital",
        "date_col": "date",
    },
    "momentum": {
        "name": "Momentum",
        "color": "#f97316",
        "trade_csv": BASE_DIR / "momentum" / "paper_trades" / "momentum_trade_log.csv",
        "daily_csv": BASE_DIR / "momentum" / "paper_trades" / "momentum_daily_summary.csv",
        "state_file": BASE_DIR / "momentum" / "paper_trades" / "momentum_state.json",
        "log_file": BASE_DIR / "momentum" / "paper_trades" / "momentum_paper.log",
        "pnl_col": "net_pnl",
        "capital_col": "capital",
        "date_col": "entry_time",
    },
    "ironcondor": {
        "name": "Iron Condor",
        "color": "#14b8a6",
        "trade_csv": BASE_DIR / "ironcondor" / "paper_trades" / "ic_trade_log.csv",
        "daily_csv": BASE_DIR / "ironcondor" / "paper_trades" / "ic_daily_summary.csv",
        "state_file": BASE_DIR / "ironcondor" / "paper_trades" / "ic_state.json",
        "log_file": BASE_DIR / "ironcondor" / "paper_trades" / "ic_paper.log",
        "pnl_col": "net_pnl",
        "capital_col": "capital",
        "date_col": "date",
    },
}

INITIAL_CAPITALS = {
    "ema": 300000,
    "sapphire": 250000,
    "momentum": 250000,
    "ironcondor": 200000,
}


def _read_csv_safe(path: Path) -> pd.DataFrame:
    """Read CSV safely, return empty DataFrame on error."""
    try:
        if path.exists() and path.stat().st_size > 0:
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()


def _read_state(path: Path) -> dict:
    """Read state JSON safely."""
    try:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _read_last_log_lines(path: Path, n: int = 5) -> list:
    """Read last N lines of a log file."""
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
                return [l.strip() for l in lines[-n:]]
    except Exception:
        pass
    return []


def get_dashboard_data() -> dict:
    """Aggregate all paper trading data for the dashboard."""
    today_str = str(date.today())
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    strategies = {}
    total_capital = 0
    total_initial = 0
    total_today_pnl = 0
    total_all_time_pnl = 0
    total_trades = 0
    total_today_trades = 0

    for key, cfg in STRATEGIES.items():
        init_cap = INITIAL_CAPITALS.get(key, 200000)
        total_initial += init_cap

        # Read trade log
        df = _read_csv_safe(cfg["trade_csv"])
        state = _read_state(cfg["state_file"])
        log_lines = _read_last_log_lines(cfg["log_file"], n=3)

        # Capital from state or CSV
        capital = state.get("capital", init_cap)
        if not df.empty and cfg["capital_col"] in df.columns:
            capital = df[cfg["capital_col"]].iloc[-1]
        total_capital += capital

        # All-time P&L
        all_time_pnl = capital - init_cap
        if not df.empty and cfg["pnl_col"] in df.columns:
            all_time_pnl = df[cfg["pnl_col"]].sum()
        total_all_time_pnl += all_time_pnl

        # Today's trades
        today_trades = pd.DataFrame()
        if not df.empty and cfg["date_col"] in df.columns:
            date_series = df[cfg["date_col"]].astype(str)
            mask = date_series.str.startswith(today_str)
            today_trades = df[mask]

        today_pnl = 0
        today_count = 0
        today_wins = 0
        if not today_trades.empty and cfg["pnl_col"] in today_trades.columns:
            today_pnl = today_trades[cfg["pnl_col"]].sum()
            today_count = len(today_trades)
            today_wins = (today_trades[cfg["pnl_col"]] > 0).sum()
        total_today_pnl += today_pnl
        total_today_trades += today_count

        # Trade count
        trade_count = len(df) if not df.empty else 0
        total_trades += trade_count

        # Win rate
        wins = 0
        if not df.empty and cfg["pnl_col"] in df.columns:
            wins = (df[cfg["pnl_col"]] > 0).sum()
        win_rate = (wins / trade_count * 100) if trade_count > 0 else 0

        # Recent trades (last 10)
        recent = []
        if not df.empty:
            for _, row in df.tail(10).iterrows():
                trade = {}
                for col in df.columns:
                    val = row[col]
                    if pd.isna(val):
                        trade[col] = None
                    elif isinstance(val, (float, int)):
                        trade[col] = round(float(val), 2)
                    else:
                        trade[col] = str(val)
                recent.append(trade)
            recent.reverse()  # newest first

        # Open position from state
        open_position = None
        ct = state.get("current_trade") or state.get("current_position")
        if ct:
            open_position = ct

        strategies[key] = {
            "name": cfg["name"],
            "color": cfg["color"],
            "capital": round(capital, 2),
            "initial_capital": init_cap,
            "all_time_pnl": round(all_time_pnl, 2),
            "all_time_return_pct": round(all_time_pnl / init_cap * 100, 2) if init_cap else 0,
            "today_pnl": round(today_pnl, 2),
            "today_trades": today_count,
            "today_wins": today_wins,
            "today_win_rate": round(today_wins / today_count * 100, 1) if today_count else 0,
            "total_trades": trade_count,
            "win_rate": round(win_rate, 1),
            "recent_trades": recent,
            "open_position": open_position,
            "last_log": log_lines,
            "is_running": _is_algo_running(key),
        }

    return {
        "timestamp": now_str,
        "date": today_str,
        "total_capital": round(total_capital, 2),
        "total_initial": total_initial,
        "total_today_pnl": round(total_today_pnl, 2),
        "total_all_time_pnl": round(total_all_time_pnl, 2),
        "total_all_time_return_pct": round(total_all_time_pnl / total_initial * 100, 2) if total_initial else 0,
        "total_trades": total_trades,
        "total_today_trades": total_today_trades,
        "strategies": strategies,
    }


def _is_algo_running(algo_key: str) -> bool:
    """Check if algo process is running via PID file."""
    pid_file = BASE_DIR / ".algo_pids.json"
    if not pid_file.exists():
        return False
    try:
        with open(pid_file) as f:
            pids = json.load(f)
        pid = pids.get(algo_key)
        if not pid:
            return False
        # Check if process alive
        if sys.platform == "win32":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x0400, False, int(pid))
            if handle:
                exit_code = ctypes.c_ulong()
                kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
                kernel32.CloseHandle(handle)
                return exit_code.value == 259
            return False
        else:
            os.kill(int(pid), 0)
            return True
    except Exception:
        return False


# ── HTTP Handler ─────────────────────────────────────────────
class DashboardHandler(SimpleHTTPRequestHandler):
    """Serves dashboard HTML and JSON API."""

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/api/data":
            self._serve_json()
        elif parsed.path in ("/", "/dashboard", "/index.html"):
            self._serve_dashboard()
        else:
            # Serve files from BASE_DIR
            self.directory = str(BASE_DIR)
            super().do_GET()

    def _serve_json(self):
        """Serve dashboard data as JSON."""
        data = get_dashboard_data()
        body = json.dumps(data, indent=2, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_dashboard(self):
        """Serve the dashboard HTML page."""
        html = DASHBOARD_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def log_message(self, format, *args):
        """Suppress request logs."""
        pass


# ── Dashboard HTML ───────────────────────────────────────────
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Algo Paper Trading — Live Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0d1117;color:#e6edf3;padding:20px;max-width:1400px;margin:0 auto}
.hdr{text-align:center;padding:16px 0;border-bottom:1px solid #30363d;margin-bottom:20px}
.hdr h1{color:#58a6ff;font-size:22px}
.hdr .sub{color:#8b949e;font-size:13px;margin-top:4px}
.hdr .live-dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#00c853;margin-right:6px;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.top-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:20px}
.tcard{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;text-align:center}
.tcard .lbl{color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:.5px}
.tcard .val{font-size:24px;font-weight:700;margin-top:4px}
.tcard .sub{color:#8b949e;font-size:12px;margin-top:2px}
.strats{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px;margin-bottom:20px}
.strat{background:#161b22;border:1px solid #30363d;border-radius:10px;overflow:hidden}
.strat-hdr{padding:14px 16px;border-bottom:1px solid #21262d;display:flex;justify-content:space-between;align-items:center}
.strat-hdr h3{font-size:14px;font-weight:600}
.strat-hdr .badge{padding:3px 8px;border-radius:12px;font-size:11px;font-weight:600}
.strat-hdr .badge.running{background:#00c85322;color:#00c853;border:1px solid #00c85344}
.strat-hdr .badge.stopped{background:#ff174422;color:#ff1744;border:1px solid #ff174444}
.strat-body{padding:16px}
.sg{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:12px}
.si{text-align:center;padding:8px;background:#0d1117;border-radius:6px}
.si .l{color:#8b949e;font-size:10px;text-transform:uppercase}
.si .v{font-size:16px;font-weight:700;margin-top:2px}
.open-pos{background:#1c2128;border:1px solid #30363d;border-radius:8px;padding:12px;margin-top:10px}
.open-pos h4{color:#f0883e;font-size:12px;margin-bottom:6px;text-transform:uppercase;letter-spacing:.5px}
.open-pos .det{color:#8b949e;font-size:12px;line-height:1.6}
.sect{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;margin-bottom:16px}
.sect h2{color:#58a6ff;font-size:15px;margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid #30363d}
table{width:100%;border-collapse:collapse;font-size:12px}
th{text-align:left;color:#8b949e;padding:6px 10px;border-bottom:1px solid #30363d;font-weight:600;white-space:nowrap}
td{padding:6px 10px;border-bottom:1px solid #21262d;white-space:nowrap}
tr:hover{background:#1c2128}
.pnl-pos{color:#00c853}.pnl-neg{color:#ff1744}.pnl-zero{color:#8b949e}
.log-lines{font-family:'Cascadia Code','Fira Code',monospace;font-size:11px;color:#8b949e;background:#0d1117;padding:8px;border-radius:6px;margin-top:8px;line-height:1.5;max-height:60px;overflow:hidden}
.footer{text-align:center;color:#484f58;font-size:11px;padding:16px 0}
.refresh-bar{position:fixed;top:0;left:0;height:2px;background:#1f6feb;transition:width 0.5s linear;z-index:999}
@media(max-width:700px){.strats{grid-template-columns:1fr}.top-cards{grid-template-columns:repeat(2,1fr)}body{padding:10px}}
</style>
</head>
<body>
<div class="refresh-bar" id="refreshBar"></div>

<div class="hdr">
<h1><span class="live-dot"></span> Algo Paper Trading — Live Dashboard</h1>
<div class="sub" id="subtitle">Loading...</div>
</div>

<div class="top-cards" id="topCards"></div>
<div class="strats" id="stratCards"></div>

<div class="sect">
<h2>Recent Trades (All Strategies)</h2>
<div style="overflow-x:auto"><table id="tradeTable"><thead><tr>
<th>Strategy</th><th>Date</th><th>Type</th><th>Entry</th><th>Exit</th><th>P&L</th><th>Capital</th><th>Status</th>
</tr></thead><tbody id="tradeBody"></tbody></table></div>
</div>

<div class="footer" id="footerText">Auto-refreshes every 30 seconds</div>

<script>
const API = '/api/data';
const REFRESH_MS = 30000;
let refreshTimer = null;
let elapsed = 0;

function fmt(n) { return n == null ? '—' : '₹' + Number(n).toLocaleString('en-IN', {maximumFractionDigits:0}); }
function fmtPct(n) { return n == null ? '—' : (n >= 0 ? '+' : '') + n.toFixed(2) + '%'; }
function pnlCls(n) { return n > 0 ? 'pnl-pos' : n < 0 ? 'pnl-neg' : 'pnl-zero'; }
function pnlIcon(n) { return n > 0 ? '▲' : n < 0 ? '▼' : '─'; }
function pnlFmt(n) { return n == null ? '—' : (n>=0?'+':'') + '₹' + Number(n).toLocaleString('en-IN',{maximumFractionDigits:0}); }

function render(d) {
  document.getElementById('subtitle').innerHTML =
    'Paper Trading | ' + d.date + ' | Updated ' + d.timestamp +
    ' | <span style="color:#58a6ff">' + d.total_today_trades + ' trades today</span>';

  // Top cards
  let tc = '';
  tc += card('Total Capital', fmt(d.total_capital), 'Started: ' + fmt(d.total_initial), '#fff');
  tc += card("Today's P&L", pnlIcon(d.total_today_pnl)+' '+pnlFmt(d.total_today_pnl),
             d.total_today_trades + ' trades', pnlCl(d.total_today_pnl));
  tc += card('All-Time P&L', pnlFmt(d.total_all_time_pnl), fmtPct(d.total_all_time_return_pct),
             pnlCl(d.total_all_time_pnl));
  tc += card('Total Trades', d.total_trades, 'across 4 strategies', '#58a6ff');
  document.getElementById('topCards').innerHTML = tc;

  // Strategy cards
  let sc = '';
  for (let [key, s] of Object.entries(d.strategies)) {
    sc += stratCard(key, s);
  }
  document.getElementById('stratCards').innerHTML = sc;

  // Recent trades table
  let allTrades = [];
  for (let [key, s] of Object.entries(d.strategies)) {
    (s.recent_trades || []).forEach(t => {
      t._strategy = s.name;
      t._color = s.color;
      allTrades.push(t);
    });
  }
  // Sort by date desc (use whatever date col is available)
  allTrades.sort((a,b) => {
    let da = a.entry_time || a.date || '';
    let db = b.entry_time || b.date || '';
    return db.localeCompare(da);
  });
  let tbody = '';
  allTrades.slice(0, 30).forEach(t => {
    let pnl = t.net_pnl || 0;
    let dateStr = (t.entry_time || t.date || '').substring(0,16);
    let typeStr = t.direction || t.exit_reason || t.option_type || '—';
    let entry = t.entry_price || t.entry_spot || t.entry_premium || '—';
    let exit = t.exit_price || t.exit_spot || t.exit_premium || '—';
    if (typeof entry === 'number') entry = entry.toFixed(1);
    if (typeof exit === 'number') exit = exit.toFixed(1);
    let status = t.status || t.exit_reason || '—';
    let cap = t.capital ? fmt(t.capital) : '—';
    tbody += '<tr><td style="color:'+t._color+'">'+t._strategy+'</td><td>'+dateStr+'</td>'
      + '<td>'+typeStr+'</td><td>'+entry+'</td><td>'+exit+'</td>'
      + '<td class="'+pnlCls(pnl)+'">'+pnlFmt(pnl)+'</td>'
      + '<td>'+cap+'</td><td>'+status+'</td></tr>';
  });
  if (!tbody) tbody = '<tr><td colspan="8" style="text-align:center;color:#8b949e;padding:20px">No trades yet — strategies will start trading tomorrow</td></tr>';
  document.getElementById('tradeBody').innerHTML = tbody;
}

function card(lbl, val, sub, color) {
  return '<div class="tcard"><div class="lbl">'+lbl+'</div>'
    + '<div class="val" style="color:'+color+'">'+val+'</div>'
    + '<div class="sub">'+sub+'</div></div>';
}

function pnlCl(n) { return n > 0 ? '#00c853' : n < 0 ? '#ff1744' : '#fff'; }

function stratCard(key, s) {
  let badge = s.is_running
    ? '<span class="badge running">RUNNING</span>'
    : '<span class="badge stopped">STOPPED</span>';

  let sg = '';
  sg += si('Capital', fmt(s.capital));
  sg += si("Today P&L", pnlFmt(s.today_pnl), pnlCl(s.today_pnl));
  sg += si('Today Trades', s.today_trades);
  sg += si('All-Time P&L', pnlFmt(s.all_time_pnl), pnlCl(s.all_time_pnl));
  sg += si('Return', fmtPct(s.all_time_return_pct), pnlCl(s.all_time_return_pct));
  sg += si('Win Rate', s.win_rate + '%', s.win_rate >= 50 ? '#00c853' : '#ff1744');

  let openPos = '';
  if (s.open_position) {
    let p = s.open_position;
    let det = '';
    if (p.direction) det += 'Direction: ' + p.direction + '<br>';
    if (p.entry_price) det += 'Entry: ' + Number(p.entry_price).toFixed(1) + '<br>';
    if (p.sl_price) det += 'SL: ' + Number(p.sl_price).toFixed(1) + '<br>';
    if (p.entry_spot) det += 'Spot: ' + Number(p.entry_spot).toFixed(1) + '<br>';
    if (p.net_credit) det += 'Credit: ' + Number(p.net_credit).toFixed(1) + '<br>';
    if (p.option_type) det += p.option_type + ' ' + (p.strike||'') + '<br>';
    openPos = '<div class="open-pos"><h4>Open Position</h4><div class="det">' + det + '</div></div>';
  }

  let logHtml = '';
  if (s.last_log && s.last_log.length > 0) {
    logHtml = '<div class="log-lines">' + s.last_log.join('<br>') + '</div>';
  }

  return '<div class="strat"><div class="strat-hdr" style="border-top:3px solid '+s.color+'">'
    + '<h3 style="color:'+s.color+'">'+s.name+'</h3>' + badge + '</div>'
    + '<div class="strat-body"><div class="sg">' + sg + '</div>'
    + openPos + logHtml + '</div></div>';
}

function si(l, v, color) {
  color = color || '#e6edf3';
  return '<div class="si"><div class="l">'+l+'</div><div class="v" style="color:'+color+'">'+v+'</div></div>';
}

async function fetchData() {
  try {
    const resp = await fetch(API);
    const data = await resp.json();
    render(data);
  } catch(e) {
    document.getElementById('subtitle').textContent = 'Error loading data: ' + e.message;
  }
}

function startRefreshBar() {
  elapsed = 0;
  const bar = document.getElementById('refreshBar');
  clearInterval(refreshTimer);
  refreshTimer = setInterval(() => {
    elapsed += 500;
    bar.style.width = (elapsed / REFRESH_MS * 100) + '%';
    if (elapsed >= REFRESH_MS) {
      bar.style.width = '0%';
      elapsed = 0;
      fetchData();
    }
  }, 500);
}

fetchData();
startRefreshBar();
</script>
</body></html>"""


# ── Main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Live Paper Trading Dashboard")
    parser.add_argument("--port", type=int, default=8050, help="Port (default: 8050)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    args = parser.parse_args()

    server = HTTPServer(("0.0.0.0", args.port), DashboardHandler)
    url = f"http://localhost:{args.port}"

    print("=" * 60)
    print("  ALGO PAPER TRADING — LIVE DASHBOARD")
    print("=" * 60)
    print(f"  URL:      {url}")
    print(f"  API:      {url}/api/data")
    print(f"  Refresh:  Every 30 seconds")
    print(f"  Press Ctrl+C to stop")
    print("=" * 60)

    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Dashboard stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
