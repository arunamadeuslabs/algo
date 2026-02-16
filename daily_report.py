"""
Daily Report & Email Notifier
==============================
Sends a daily email summary of paper trading results after market close.
Also generates an HTML dashboard file viewable from any browser.

Usage:
  python daily_report.py                    # Generate report + send email
  python daily_report.py --no-email         # Just generate HTML dashboard
  python daily_report.py --test             # Send test email to verify setup

Setup:
  Set these in your .env file:
    EMAIL_SENDER=your.email@gmail.com
    EMAIL_PASSWORD=your_app_password        # Gmail App Password (not regular password)
    EMAIL_RECEIVER=your.email@gmail.com     # Where to receive reports

  For Gmail:
    1. Go to myaccount.google.com -> Security -> 2-Step Verification (enable)
    2. Search 'App passwords' -> Create one for 'Mail'
    3. Use that 16-char password as EMAIL_PASSWORD
"""

import os
import sys
import json
import smtplib
import argparse
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, date, timedelta
from pathlib import Path

import pandas as pd

# -- Paths --
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
BACKTEST_DIR = BASE_DIR / "backtest"
SAPPHIRE_DIR = BASE_DIR / "sapphire"
DASHBOARD_FILE = BASE_DIR / "dashboard.html"

# Trade log paths
EMA_TRADE_LOG = BACKTEST_DIR / "paper_trades" / "paper_trade_log.csv"
EMA_DAILY_LOG = BACKTEST_DIR / "paper_trades" / "daily_summary.csv"
EMA_STATE = BACKTEST_DIR / "paper_trades" / "state.json"

SAPPHIRE_TRADE_LOG = SAPPHIRE_DIR / "paper_trades" / "sapphire_trade_log.csv"
SAPPHIRE_DAILY_LOG = SAPPHIRE_DIR / "paper_trades" / "sapphire_daily_summary.csv"
SAPPHIRE_STATE = SAPPHIRE_DIR / "paper_trades" / "sapphire_state.json"

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

# Email config from env
EMAIL_SENDER = os.environ.get("EMAIL_SENDER", "")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "")
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER", "")
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))


# -- Data Loading --
def load_trades(csv_path: Path) -> pd.DataFrame:
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                return df
        except Exception:
            pass
    return pd.DataFrame()


def load_state(state_path: Path) -> dict:
    if state_path.exists():
        try:
            with open(state_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def get_today_str():
    return str(date.today())


def filter_today(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    today = get_today_str()
    if df.empty or date_col not in df.columns:
        return pd.DataFrame()
    return df[df[date_col].astype(str).str.startswith(today)]


def calc_stats(df: pd.DataFrame, pnl_col: str = "net_pnl") -> dict:
    if df.empty or pnl_col not in df.columns:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
                "total_pnl": 0, "avg_pnl": 0, "best": 0, "worst": 0}
    pnl = df[pnl_col].astype(float)
    wins = (pnl > 0).sum()
    losses = (pnl <= 0).sum()
    total = len(pnl)
    return {
        "trades": total, "wins": int(wins), "losses": int(losses),
        "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
        "total_pnl": round(pnl.sum(), 2),
        "avg_pnl": round(pnl.mean(), 2) if total > 0 else 0,
        "best": round(pnl.max(), 2) if total > 0 else 0,
        "worst": round(pnl.min(), 2) if total > 0 else 0,
    }


# -- Report Generation --
def generate_report() -> dict:
    report = {
        "date": get_today_str(),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ema": {}, "sapphire": {},
    }

    # EMA Crossover
    ema_all = load_trades(EMA_TRADE_LOG)
    ema_state = load_state(EMA_STATE)
    ema_today = filter_today(ema_all, "entry_time") if not ema_all.empty else pd.DataFrame()
    report["ema"] = {
        "today": calc_stats(ema_today),
        "all_time": calc_stats(ema_all),
        "capital": ema_state.get("capital", 500000),
        "initial_capital": ema_state.get("initial_capital", 500000),
        "total_trades": len(ema_all),
        "today_trades_detail": ema_today.to_dict("records") if not ema_today.empty else [],
    }

    # Sapphire
    sap_all = load_trades(SAPPHIRE_TRADE_LOG)
    sap_state = load_state(SAPPHIRE_STATE)
    sap_today = filter_today(sap_all, "date") if not sap_all.empty else pd.DataFrame()
    report["sapphire"] = {
        "today": calc_stats(sap_today),
        "all_time": calc_stats(sap_all),
        "capital": sap_state.get("capital", 150000),
        "initial_capital": sap_state.get("initial_capital", 150000),
        "total_trades": len(sap_all),
        "today_trades_detail": sap_today.to_dict("records") if not sap_today.empty else [],
    }

    # Combined
    ema_cap = report["ema"]["capital"]
    sap_cap = report["sapphire"]["capital"]
    ema_init = report["ema"]["initial_capital"]
    sap_init = report["sapphire"]["initial_capital"]
    report["combined"] = {
        "total_capital": round(ema_cap + sap_cap, 2),
        "initial_capital": round(ema_init + sap_init, 2),
        "total_return": round((ema_cap + sap_cap) - (ema_init + sap_init), 2),
        "total_return_pct": round(((ema_cap + sap_cap) / (ema_init + sap_init) - 1) * 100, 2),
        "today_pnl": round(report["ema"]["today"]["total_pnl"] + report["sapphire"]["today"]["total_pnl"], 2),
        "today_trades": report["ema"]["today"]["trades"] + report["sapphire"]["today"]["trades"],
    }
    return report


# -- HTML Dashboard --
def generate_dashboard(report: dict) -> str:
    def pnl_color(val):
        if val > 0: return "#00c853"
        elif val < 0: return "#ff1744"
        return "#ffffff"

    def pnl_icon(val):
        if val > 0: return "▲"
        elif val < 0: return "▼"
        return "─"

    def format_inr(val):
        return f"₹{val:+,.2f}" if val != 0 else "₹0.00"

    ema = report["ema"]
    sap = report["sapphire"]
    comb = report["combined"]

    ema_rows = ""
    for t in ema.get("today_trades_detail", []):
        pnl = t.get("net_pnl", t.get("pnl", 0))
        ema_rows += f'<tr><td>{t.get("entry_time","N/A")[:16]}</td><td>{t.get("type",t.get("signal","N/A"))}</td><td>{t.get("entry_price","N/A")}</td><td>{t.get("exit_price","N/A")}</td><td style="color:{pnl_color(pnl)};font-weight:bold">{format_inr(pnl)}</td><td>{t.get("exit_reason","N/A")}</td></tr>'

    sap_rows = ""
    for t in sap.get("today_trades_detail", []):
        pnl = t.get("net_pnl", 0)
        sap_rows += f'<tr><td>{t.get("date","N/A")}</td><td>CE:{t.get("ce_strike","?")} / PE:{t.get("pe_strike","?")}</td><td>{t.get("entry_spot","N/A")}</td><td>{t.get("exit_spot","N/A")}</td><td style="color:{pnl_color(pnl)};font-weight:bold">{format_inr(pnl)}</td><td>{t.get("exit_reason","N/A")}</td></tr>'

    daily_chart_data = []
    for log_path, label in [(EMA_DAILY_LOG, "EMA"), (SAPPHIRE_DAILY_LOG, "Sapphire")]:
        if log_path.exists():
            try:
                df = pd.read_csv(log_path)
                for _, row in df.tail(7).iterrows():
                    daily_chart_data.append({"date": str(row.get("date","")), "pnl": float(row.get("day_pnl",0)), "strategy": label})
            except Exception:
                pass

    daily_rows = ""
    for d in daily_chart_data[-14:]:
        pnl = d["pnl"]
        bar_width = min(abs(pnl) / 100, 200)
        bar_color = "#00c853" if pnl >= 0 else "#ff1744"
        daily_rows += f'<tr><td>{d["date"]}</td><td>{d["strategy"]}</td><td style="color:{pnl_color(pnl)};font-weight:bold">{format_inr(pnl)}</td><td><div style="background:{bar_color};height:16px;width:{bar_width}px;border-radius:3px;display:inline-block"></div></td></tr>'

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta http-equiv="refresh" content="300">
<title>Nifty Algo Dashboard</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0d1117;color:#e6edf3;padding:20px;max-width:1200px;margin:0 auto}}
.header{{text-align:center;padding:20px 0;border-bottom:1px solid #30363d;margin-bottom:24px}}
.header h1{{color:#58a6ff;font-size:24px}}
.header .subtitle{{color:#8b949e;font-size:14px;margin-top:4px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;margin-bottom:24px}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px}}
.card h3{{color:#8b949e;font-size:12px;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}}
.card .value{{font-size:28px;font-weight:700}}
.card .sub{{color:#8b949e;font-size:13px;margin-top:4px}}
.section{{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px;margin-bottom:20px}}
.section h2{{color:#58a6ff;font-size:16px;margin-bottom:16px;padding-bottom:8px;border-bottom:1px solid #30363d}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{text-align:left;color:#8b949e;padding:8px 12px;border-bottom:1px solid #30363d;font-weight:600}}
td{{padding:8px 12px;border-bottom:1px solid #21262d}}
tr:hover{{background:#1c2128}}
.stat-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px}}
.stat-item{{text-align:center;padding:12px;background:#0d1117;border-radius:8px}}
.stat-item .label{{color:#8b949e;font-size:11px;text-transform:uppercase}}
.stat-item .val{{font-size:20px;font-weight:700;margin-top:4px}}
.footer{{text-align:center;color:#484f58;font-size:12px;padding:20px 0}}
@media(max-width:600px){{.cards{{grid-template-columns:1fr}}.stat-grid{{grid-template-columns:repeat(2,1fr)}}body{{padding:12px}}}}
</style>
</head>
<body>
<div class="header">
<h1>Nifty Algo Trading Dashboard</h1>
<div class="subtitle">Paper Trading | {report["date"]} | Updated {report["generated_at"]}</div>
</div>
<div class="cards">
<div class="card">
<h3>Today's P&amp;L</h3>
<div class="value" style="color:{pnl_color(comb["today_pnl"])}">{pnl_icon(comb["today_pnl"])} {format_inr(comb["today_pnl"])}</div>
<div class="sub">{comb["today_trades"]} trades today</div>
</div>
<div class="card">
<h3>Total Capital</h3>
<div class="value" style="color:{pnl_color(comb["total_return"])}">₹{comb["total_capital"]:,.0f}</div>
<div class="sub">Started: ₹{comb["initial_capital"]:,.0f}</div>
</div>
<div class="card">
<h3>All-Time Return</h3>
<div class="value" style="color:{pnl_color(comb["total_return"])}">{pnl_icon(comb["total_return"])} {comb["total_return_pct"]:+.2f}%</div>
<div class="sub">{format_inr(comb["total_return"])} net profit</div>
</div>
</div>
<div class="cards">
<div class="section" style="margin-bottom:0">
<h2>EMA Crossover Strategy</h2>
<div class="stat-grid">
<div class="stat-item"><div class="label">Capital</div><div class="val">₹{ema["capital"]:,.0f}</div></div>
<div class="stat-item"><div class="label">Today P&amp;L</div><div class="val" style="color:{pnl_color(ema["today"]["total_pnl"])}">{format_inr(ema["today"]["total_pnl"])}</div></div>
<div class="stat-item"><div class="label">Total Trades</div><div class="val">{ema["total_trades"]}</div></div>
<div class="stat-item"><div class="label">Win Rate</div><div class="val">{ema["all_time"]["win_rate"]}%</div></div>
<div class="stat-item"><div class="label">All-Time P&amp;L</div><div class="val" style="color:{pnl_color(ema["all_time"]["total_pnl"])}">{format_inr(ema["all_time"]["total_pnl"])}</div></div>
<div class="stat-item"><div class="label">Avg Trade</div><div class="val" style="color:{pnl_color(ema["all_time"]["avg_pnl"])}">{format_inr(ema["all_time"]["avg_pnl"])}</div></div>
</div></div>
<div class="section" style="margin-bottom:0">
<h2>Sapphire Short Strangle</h2>
<div class="stat-grid">
<div class="stat-item"><div class="label">Capital</div><div class="val">₹{sap["capital"]:,.0f}</div></div>
<div class="stat-item"><div class="label">Today P&amp;L</div><div class="val" style="color:{pnl_color(sap["today"]["total_pnl"])}">{format_inr(sap["today"]["total_pnl"])}</div></div>
<div class="stat-item"><div class="label">Total Trades</div><div class="val">{sap["total_trades"]}</div></div>
<div class="stat-item"><div class="label">Win Rate</div><div class="val">{sap["all_time"]["win_rate"]}%</div></div>
<div class="stat-item"><div class="label">All-Time P&amp;L</div><div class="val" style="color:{pnl_color(sap["all_time"]["total_pnl"])}">{format_inr(sap["all_time"]["total_pnl"])}</div></div>
<div class="stat-item"><div class="label">Avg Trade</div><div class="val" style="color:{pnl_color(sap["all_time"]["avg_pnl"])}">{format_inr(sap["all_time"]["avg_pnl"])}</div></div>
</div></div>
</div>
<div class="section">
<h2>Today's Trades - EMA Crossover</h2>
{"<table><tr><th>Time</th><th>Type</th><th>Entry</th><th>Exit</th><th>P&L</th><th>Exit Reason</th></tr>" + ema_rows + "</table>" if ema_rows else '<p style="color:#8b949e;text-align:center;padding:20px">No trades today</p>'}
</div>
<div class="section">
<h2>Today's Trades - Sapphire Strangle</h2>
{"<table><tr><th>Date</th><th>Strikes</th><th>Entry Spot</th><th>Exit Spot</th><th>P&L</th><th>Exit Reason</th></tr>" + sap_rows + "</table>" if sap_rows else '<p style="color:#8b949e;text-align:center;padding:20px">No trades today</p>'}
</div>
<div class="section">
<h2>Recent Daily P&amp;L</h2>
{"<table><tr><th>Date</th><th>Strategy</th><th>P&L</th><th>Bar</th></tr>" + daily_rows + "</table>" if daily_rows else '<p style="color:#8b949e;text-align:center;padding:20px">No history yet - trades will appear after the first trading day</p>'}
</div>
<div class="footer">Nifty Algo Trader - Paper Trading Dashboard - Auto-refreshes every 5 minutes</div>
</body></html>'''
    return html


# -- Email --
def generate_email_body(report: dict) -> str:
    ema = report["ema"]
    sap = report["sapphire"]
    comb = report["combined"]
    def clr(val):
        return "#00c853" if val > 0 else "#ff1744" if val < 0 else "#888888"

    return f'''
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:auto;background:#1a1a2e;color:#eee;padding:24px;border-radius:12px">
        <h2 style="color:#58a6ff;text-align:center;margin-bottom:4px">Nifty Algo - Daily Report</h2>
        <p style="text-align:center;color:#888;font-size:13px">{report["date"]} | Paper Trading</p>
        <hr style="border:1px solid #333;margin:16px 0">
        <table style="width:100%;border-collapse:collapse;margin-bottom:16px">
            <tr>
                <td style="text-align:center;padding:12px;background:#16213e;border-radius:8px 0 0 8px">
                    <div style="color:#888;font-size:11px">TODAY P&amp;L</div>
                    <div style="font-size:24px;font-weight:bold;color:{clr(comb["today_pnl"])}">₹{comb["today_pnl"]:+,.2f}</div>
                    <div style="color:#888;font-size:12px">{comb["today_trades"]} trades</div>
                </td>
                <td style="text-align:center;padding:12px;background:#16213e">
                    <div style="color:#888;font-size:11px">TOTAL CAPITAL</div>
                    <div style="font-size:24px;font-weight:bold">₹{comb["total_capital"]:,.0f}</div>
                    <div style="color:#888;font-size:12px">from ₹{comb["initial_capital"]:,.0f}</div>
                </td>
                <td style="text-align:center;padding:12px;background:#16213e;border-radius:0 8px 8px 0">
                    <div style="color:#888;font-size:11px">TOTAL RETURN</div>
                    <div style="font-size:24px;font-weight:bold;color:{clr(comb["total_return"])}">{comb["total_return_pct"]:+.2f}%</div>
                    <div style="color:#888;font-size:12px">₹{comb["total_return"]:+,.2f}</div>
                </td>
            </tr>
        </table>
        <div style="background:#16213e;border-radius:8px;padding:16px;margin-bottom:12px">
            <h3 style="color:#58a6ff;margin-bottom:8px;font-size:14px">EMA Crossover</h3>
            <table style="width:100%;font-size:13px">
                <tr><td style="color:#888;padding:4px 0">Today P&amp;L</td><td style="text-align:right;color:{clr(ema["today"]["total_pnl"])};font-weight:bold">₹{ema["today"]["total_pnl"]:+,.2f}</td></tr>
                <tr><td style="color:#888;padding:4px 0">Today Trades</td><td style="text-align:right">{ema["today"]["trades"]} ({ema["today"]["wins"]}W / {ema["today"]["losses"]}L)</td></tr>
                <tr><td style="color:#888;padding:4px 0">Capital</td><td style="text-align:right">₹{ema["capital"]:,.0f}</td></tr>
                <tr><td style="color:#888;padding:4px 0">All-Time Win Rate</td><td style="text-align:right">{ema["all_time"]["win_rate"]}% ({ema["total_trades"]} trades)</td></tr>
                <tr><td style="color:#888;padding:4px 0">All-Time P&amp;L</td><td style="text-align:right;color:{clr(ema["all_time"]["total_pnl"])};font-weight:bold">₹{ema["all_time"]["total_pnl"]:+,.2f}</td></tr>
            </table>
        </div>
        <div style="background:#16213e;border-radius:8px;padding:16px;margin-bottom:12px">
            <h3 style="color:#58a6ff;margin-bottom:8px;font-size:14px">Sapphire Short Strangle</h3>
            <table style="width:100%;font-size:13px">
                <tr><td style="color:#888;padding:4px 0">Today P&amp;L</td><td style="text-align:right;color:{clr(sap["today"]["total_pnl"])};font-weight:bold">₹{sap["today"]["total_pnl"]:+,.2f}</td></tr>
                <tr><td style="color:#888;padding:4px 0">Today Trades</td><td style="text-align:right">{sap["today"]["trades"]} ({sap["today"]["wins"]}W / {sap["today"]["losses"]}L)</td></tr>
                <tr><td style="color:#888;padding:4px 0">Capital</td><td style="text-align:right">₹{sap["capital"]:,.0f}</td></tr>
                <tr><td style="color:#888;padding:4px 0">All-Time Win Rate</td><td style="text-align:right">{sap["all_time"]["win_rate"]}% ({sap["total_trades"]} trades)</td></tr>
                <tr><td style="color:#888;padding:4px 0">All-Time P&amp;L</td><td style="text-align:right;color:{clr(sap["all_time"]["total_pnl"])};font-weight:bold">₹{sap["all_time"]["total_pnl"]:+,.2f}</td></tr>
            </table>
        </div>
        <hr style="border:1px solid #333;margin:16px 0">
        <p style="text-align:center;color:#555;font-size:11px">Automated report from Nifty Algo Trader | Paper Trading Mode</p>
    </div>'''


def send_email(subject: str, html_body: str):
    if not EMAIL_SENDER or not EMAIL_PASSWORD or not EMAIL_RECEIVER:
        print("  Email not configured. Set EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER in .env")
        return False
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg.attach(MIMEText(html_body, "html"))
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        print(f"  Email sent to {EMAIL_RECEIVER}")
        return True
    except Exception as e:
        print(f"  Email failed: {e}")
        return False


# -- Main --
def main():
    parser = argparse.ArgumentParser(description="Daily Algo Trading Report")
    parser.add_argument("--no-email", action="store_true", help="Skip sending email")
    parser.add_argument("--test", action="store_true", help="Send test email")
    parser.add_argument("--dashboard-only", action="store_true", help="Only generate HTML dashboard")
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("  DAILY ALGO REPORT")
    print("=" * 50)

    report = generate_report()

    html = generate_dashboard(report)
    with open(DASHBOARD_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  Dashboard saved: {DASHBOARD_FILE}")

    if args.dashboard_only:
        print("  Done (dashboard only).")
        return

    comb = report["combined"]
    print(f"\n  Date: {report['date']}")
    print(f"  Today P&L: ₹{comb['today_pnl']:+,.2f} ({comb['today_trades']} trades)")
    print(f"  Total Capital: ₹{comb['total_capital']:,.0f}")
    print(f"  Total Return: {comb['total_return_pct']:+.2f}%")

    if args.test:
        subject = f"[TEST] Nifty Algo Report - {report['date']}"
        body = generate_email_body(report)
        send_email(subject, body)
    elif not args.no_email:
        pnl = comb["today_pnl"]
        icon = "+" if pnl > 0 else "-" if pnl < 0 else "="
        subject = f"[{icon}] Nifty Algo: Rs{pnl:+,.0f} | {report['date']}"
        body = generate_email_body(report)
        send_email(subject, body)
    else:
        print("  Email skipped (--no-email).")

    print(f"\n  Done. {report['generated_at']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
