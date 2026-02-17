"""
Visualization & Excel Report for Iron Condor Strategy.
Dark-themed charts + professional Excel workbook.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# ─── Color Palette ──────────────────────────────────────────────────────────
IC_BG = "#0a0e27"
IC_CARD = "#141938"
IC_TEAL = "#14b8a6"
IC_CYAN = "#06b6d4"
IC_GREEN = "#10b981"
IC_RED = "#ef4444"
IC_GOLD = "#f59e0b"
IC_PURPLE = "#a855f7"
IC_TEXT = "#e2e8f0"
IC_GRID = "#1e293b"


def setup_dark_style():
    """Apply Iron Condor dark theme."""
    plt.rcParams.update({
        "figure.facecolor": IC_BG,
        "axes.facecolor": IC_CARD,
        "axes.edgecolor": IC_GRID,
        "axes.labelcolor": IC_TEXT,
        "text.color": IC_TEXT,
        "xtick.color": IC_TEXT,
        "ytick.color": IC_TEXT,
        "grid.color": IC_GRID,
        "grid.alpha": 0.3,
        "font.family": "sans-serif",
        "font.size": 10,
    })


# ─── Equity Curve ───────────────────────────────────────────────────────────

def plot_equity_curve(result, save_path: str = "results/ic_equity.png"):
    """Plot equity curve with drawdown overlay."""
    setup_dark_style()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1],
                                     gridspec_kw={"hspace": 0.08})

    equity = np.array(result.equity_curve)
    x = range(len(equity))

    ax1.fill_between(x, equity, equity[0], alpha=0.15, color=IC_TEAL)
    ax1.plot(x, equity, color=IC_TEAL, linewidth=1.5, label="Equity")

    peak_idx = np.argmax(equity)
    ax1.scatter([peak_idx], [equity[peak_idx]], color=IC_GOLD, s=80, zorder=5,
                label=f"Peak: ₹{equity[peak_idx]:,.0f}")

    ax1.scatter([0], [equity[0]], color=IC_PURPLE, s=60, zorder=5)
    ax1.scatter([len(equity)-1], [equity[-1]],
                color=IC_GREEN if equity[-1] > equity[0] else IC_RED,
                s=60, zorder=5)

    ax1.set_title("IRON CONDOR — Equity Curve", fontsize=16, fontweight="bold",
                   color=IC_TEAL, pad=15)
    ax1.set_ylabel("Capital (₹)")
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.3)
    ax1.grid(True, alpha=0.2)
    ax1.set_xticklabels([])

    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak * 100

    ax2.fill_between(x, -dd, 0, alpha=0.4, color=IC_RED)
    ax2.plot(x, -dd, color=IC_RED, linewidth=0.8)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Trade #")
    ax2.grid(True, alpha=0.2)
    ax2.set_ylim([-max(dd) * 1.3 if max(dd) > 0 else -1, 0.5])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=IC_BG)
    plt.close()
    print(f"  Equity curve saved: {save_path}")


# ─── Monthly Heatmap ────────────────────────────────────────────────────────

def plot_monthly_heatmap(result, save_path: str = "results/ic_monthly.png"):
    """Plot monthly P&L heatmap."""
    setup_dark_style()

    if not result.daily_pnl:
        return

    df = pd.DataFrame(result.daily_pnl)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    monthly = df.groupby(["year", "month"])["net_pnl"].sum().unstack(fill_value=0)
    monthly_pct = (monthly / result.initial_capital) * 100

    fig, ax = plt.subplots(figsize=(14, max(4, len(monthly_pct) * 0.8)))

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    max_abs = max(abs(monthly_pct.values.max()), abs(monthly_pct.values.min()), 1)

    for i, year in enumerate(monthly_pct.index):
        for j in range(1, 13):
            val = monthly_pct.loc[year, j] if j in monthly_pct.columns else 0
            color = IC_GREEN if val > 0 else IC_RED if val < 0 else IC_GRID
            alpha = min(abs(val) / max_abs * 0.8 + 0.2, 1.0) if val != 0 else 0.1

            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                  facecolor=color, alpha=alpha, edgecolor=IC_GRID, linewidth=0.5)
            ax.add_patch(rect)

            if val != 0:
                text_color = "#ffffff" if abs(val) > max_abs * 0.3 else IC_TEXT
                ax.text(j, i, f"{val:+.1f}%", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=text_color)

    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(-0.5, len(monthly_pct) - 0.5)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels)
    ax.set_yticks(range(len(monthly_pct)))
    ax.set_yticklabels(monthly_pct.index)
    ax.set_title("IRON CONDOR — Monthly Returns (%)", fontsize=14,
                  fontweight="bold", color=IC_TEAL, pad=15)
    ax.invert_yaxis()

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=IC_BG)
    plt.close()
    print(f"  Monthly heatmap saved: {save_path}")


# ─── Trade Analysis ─────────────────────────────────────────────────────────

def plot_trade_analysis(result, save_path: str = "results/ic_trades.png"):
    """Plot trade P&L distribution and exit reasons."""
    setup_dark_style()

    if not result.trades:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    pnls = [t.net_pnl for t in result.trades]

    # 1. P&L Distribution
    ax = axes[0, 0]
    colors = [IC_GREEN if p > 0 else IC_RED for p in pnls]
    ax.bar(range(len(pnls)), pnls, color=colors, alpha=0.7, width=0.8)
    ax.axhline(0, color=IC_TEXT, linewidth=0.5, alpha=0.5)
    ax.set_title("Trade-wise P&L", fontsize=12, color=IC_TEAL)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Net P&L (₹)")
    ax.grid(True, alpha=0.2)

    # 2. P&L Histogram
    ax = axes[0, 1]
    ax.hist(pnls, bins=25, color=IC_TEAL, alpha=0.7, edgecolor=IC_GRID)
    ax.axvline(0, color=IC_RED, linewidth=1, linestyle="--", alpha=0.7)
    avg_pnl = np.mean(pnls)
    ax.axvline(avg_pnl, color=IC_GOLD, linewidth=1.5, linestyle="--",
               label=f"Avg: ₹{avg_pnl:,.0f}")
    ax.set_title("P&L Distribution", fontsize=12, color=IC_TEAL)
    ax.set_xlabel("P&L (₹)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # 3. Exit Reasons Pie
    ax = axes[1, 0]
    reasons = {}
    for t in result.trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

    if reasons:
        labels = list(reasons.keys())
        sizes = list(reasons.values())
        pie_colors = [IC_GREEN, IC_RED, IC_GOLD, IC_PURPLE, IC_CYAN, IC_TEAL][:len(labels)]

        wedges, texts, autotexts = ax.pie(sizes, labels=None, autopct="%1.0f%%",
                                           colors=pie_colors, startangle=90,
                                           textprops={"fontsize": 9, "color": IC_TEXT})
        ax.legend(labels, loc="center left", bbox_to_anchor=(-0.2, 0.5),
                   fontsize=8, framealpha=0.3)
    ax.set_title("Exit Reasons", fontsize=12, color=IC_TEAL)

    # 4. Cumulative P&L
    ax = axes[1, 1]
    cum_pnl = np.cumsum(pnls)
    ax.fill_between(range(len(cum_pnl)), cum_pnl, 0, alpha=0.15, color=IC_TEAL)
    ax.plot(cum_pnl, color=IC_TEAL, linewidth=1.5)
    ax.axhline(0, color=IC_TEXT, linewidth=0.5, alpha=0.5)
    ax.set_title("Cumulative P&L", fontsize=12, color=IC_TEAL)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cum P&L (₹)")
    ax.grid(True, alpha=0.2)

    plt.suptitle("IRON CONDOR — Trade Analysis", fontsize=16,
                  fontweight="bold", color=IC_TEAL, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=IC_BG)
    plt.close()
    print(f"  Trade analysis saved: {save_path}")


# ─── Excel Report ───────────────────────────────────────────────────────────

def save_excel_report(result, save_path: str = "results/ic_report.xlsx"):
    """Save detailed backtest report to Excel."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
        # Summary sheet
        summary = {
            "Metric": [
                "Initial Capital", "Final Capital", "Total Net P&L", "ROI %",
                "Total Trades", "Winners", "Losers", "Win Rate %",
                "Gross P&L", "Total Costs",
                "Avg Win", "Avg Loss", "Best Trade", "Worst Trade",
                "Profit Factor", "Sharpe Ratio",
                "Max Drawdown %", "Max Drawdown ₹",
                "Max Consecutive Wins", "Max Consecutive Losses",
                "Best Month %", "Worst Month %", "Avg Monthly Return %",
                "Avg Credit Collected (pts)", "Skipped Days (filters)",
            ],
            "Value": [
                result.initial_capital, result.final_capital, result.total_net_pnl,
                round(result.roi_pct, 2),
                result.total_trades, result.winners, result.losers,
                round(result.win_rate, 1),
                result.total_gross_pnl, result.total_costs,
                round(result.avg_win, 2), round(result.avg_loss, 2),
                result.best_trade, result.worst_trade,
                round(result.profit_factor, 2), round(result.sharpe_ratio, 2),
                round(result.max_drawdown_pct, 2), round(result.max_drawdown_inr, 2),
                result.max_consecutive_wins, result.max_consecutive_losses,
                round(result.best_month, 2), round(result.worst_month, 2),
                round(result.avg_monthly_return, 2),
                round(result.avg_credit_collected, 1), result.skipped_days,
            ],
        }
        pd.DataFrame(summary).to_excel(writer, sheet_name="Summary", index=False)

        # Daily P&L sheet
        if result.daily_pnl:
            df_daily = pd.DataFrame(result.daily_pnl)
            df_daily.to_excel(writer, sheet_name="Daily PnL", index=False)

        # Trade log sheet
        if result.trades:
            rows = []
            for i, t in enumerate(result.trades):
                rows.append({
                    "#": i + 1,
                    "Date": str(t.date),
                    "Entry Spot": round(t.entry_spot, 2),
                    "Sell CE": t.short_ce.strike if t.short_ce else 0,
                    "Buy CE": t.long_ce.strike if t.long_ce else 0,
                    "Sell PE": t.short_pe.strike if t.short_pe else 0,
                    "Buy PE": t.long_pe.strike if t.long_pe else 0,
                    "Net Credit": round(t.total_net_credit, 2),
                    "Gross P&L": round(t.gross_pnl, 2),
                    "Costs": round(t.total_costs, 2),
                    "Net P&L": round(t.net_pnl, 2),
                    "Exit": t.exit_reason,
                    "VIX": round(t.vix_at_entry, 2),
                    "RSI": round(t.rsi_at_entry, 2),
                })
            pd.DataFrame(rows).to_excel(writer, sheet_name="Trade Log", index=False)

    print(f"  Excel report saved: {save_path}")
