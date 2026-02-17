"""Analyze Supertrend trade log to find loss patterns."""
import pandas as pd
import numpy as np

df = pd.read_csv('supertrend/paper_trades/supertrend_trade_log.csv')
print(f"Total trades: {len(df)}")
print(f"Overall Net P&L: {df['net_pnl'].sum():.0f}")
print(f"Overall Gross P&L: {df['gross_pnl'].sum():.0f}")
print(f"Total Costs: {df['costs'].sum():.0f}")
print()

# Exit status breakdown
print("=== Exit Status Breakdown ===")
for s in df['status'].unique():
    sub = df[df['status'] == s]
    w = len(sub[sub['net_pnl'] > 0])
    l = len(sub[sub['net_pnl'] <= 0])
    print(f"  {s}: {len(sub)} trades, W={w}, L={l}, P&L={sub['net_pnl'].sum():.0f}, Avg={sub['net_pnl'].mean():.0f}")
print()

# Direction breakdown
print("=== Direction Breakdown ===")
for d in df['direction'].unique():
    sub = df[df['direction'] == d]
    w = len(sub[sub['net_pnl'] > 0])
    print(f"  {d}: {len(sub)} trades, W={w} ({w/len(sub)*100:.1f}%), P&L={sub['net_pnl'].sum():.0f}, Avg={sub['net_pnl'].mean():.0f}")
print()

# Win/Loss stats
wins = df[df['net_pnl'] > 0]
losses = df[df['net_pnl'] <= 0]
print(f"=== Win/Loss Stats ===")
print(f"  Wins: {len(wins)}, Avg Win: {wins['net_pnl'].mean():.0f}")
print(f"  Losses: {len(losses)}, Avg Loss: {losses['net_pnl'].mean():.0f}")
print(f"  Win Rate: {len(wins)/len(df)*100:.1f}%")
print(f"  Expectancy: {df['net_pnl'].mean():.0f} per trade")
print()

# Max favorable analysis (trades that went positive but ended negative)
reversals = df[(df['max_favorable'] > 10) & (df['net_pnl'] <= 0)]
print(f"=== Reversal Analysis (went >10pts favorable, still lost) ===")
print(f"  Count: {len(reversals)}")
if len(reversals) > 0:
    print(f"  Avg max favorable: {reversals['max_favorable'].mean():.1f} pts")
    print(f"  Avg loss: {reversals['net_pnl'].mean():.0f}")
    print(f"  Total lost: {reversals['net_pnl'].sum():.0f}")
print()

# SL distance analysis
df['sl_distance'] = abs(df['entry_price'] - df['sl_price'])
df['target_distance'] = abs(df['target_price'] - df['entry_price'])
df['actual_move'] = abs(df['exit_price'] - df['entry_price'])
print(f"=== SL/Target Analysis ===")
print(f"  Avg SL distance: {df['sl_distance'].mean():.1f} pts")
print(f"  Avg Target distance: {df['target_distance'].mean():.1f} pts")
print(f"  Avg actual move: {df['actual_move'].mean():.1f} pts")
print(f"  SL hits with high max_favorable: {len(df[(df['status']=='SL_HIT') & (df['max_favorable']>15)])}")
print()

# TIME_EXIT analysis
time_exits = df[df['status'] == 'TIME_EXIT']
if len(time_exits) > 0:
    print(f"=== TIME_EXIT Analysis ===")
    te_wins = len(time_exits[time_exits['net_pnl'] > 0])
    te_losses = len(time_exits[time_exits['net_pnl'] <= 0])
    print(f"  Count: {len(time_exits)}, W={te_wins}, L={te_losses}")
    print(f"  Total P&L: {time_exits['net_pnl'].sum():.0f}")
    print(f"  Avg P&L: {time_exits['net_pnl'].mean():.0f}")
    print()

# P&L distribution
print(f"=== P&L Distribution ===")
print(f"  Min: {df['net_pnl'].min():.0f}")
print(f"  25%: {df['net_pnl'].quantile(0.25):.0f}")
print(f"  50%: {df['net_pnl'].quantile(0.50):.0f}")
print(f"  75%: {df['net_pnl'].quantile(0.75):.0f}")
print(f"  Max: {df['net_pnl'].max():.0f}")
print()

# Gross P&L distribution (before costs)
print(f"=== Gross P&L Distribution ===")
print(f"  Min: {df['gross_pnl'].min():.0f}")
print(f"  Median: {df['gross_pnl'].median():.0f}")
print(f"  Max: {df['gross_pnl'].max():.0f}")
print(f"  Gross profit: {df[df['gross_pnl']>0]['gross_pnl'].sum():.0f}")
print(f"  Gross loss: {df[df['gross_pnl']<=0]['gross_pnl'].sum():.0f}")
print()

# Daily P&L
df['date'] = pd.to_datetime(df['date'])
daily = df.groupby(df['date'].dt.date).agg(
    trades=('net_pnl', 'count'),
    pnl=('net_pnl', 'sum')
)
print(f"=== Daily Stats ===")
print(f"  Trading days: {len(daily)}")
print(f"  Profitable days: {len(daily[daily['pnl']>0])}")
print(f"  Loss days: {len(daily[daily['pnl']<=0])}")
print(f"  Avg daily P&L: {daily['pnl'].mean():.0f}")
print(f"  Avg trades/day: {daily['trades'].mean():.1f}")
print(f"  Best day: {daily['pnl'].max():.0f}")
print(f"  Worst day: {daily['pnl'].min():.0f}")
