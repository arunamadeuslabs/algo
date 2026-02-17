#!/bin/bash
# ============================================================
#  Nifty Algo Trader — GCP Deploy & Backtest Runner
#  Run this on your GCP VM after git pull to populate trade data
#
#  Usage:
#    bash deploy_gcp.sh          # Full deploy (pull + backtest + regenerate)
#    bash deploy_gcp.sh --skip-pull   # Skip git pull, just run backtests
# ============================================================

set -e

ALGO_DIR="$HOME/algo"
PYTHON="$ALGO_DIR/.venv/bin/python3"

# Check if --skip-pull flag is passed
SKIP_PULL=false
if [ "$1" = "--skip-pull" ]; then
    SKIP_PULL=true
fi

echo ""
echo "============================================"
echo "  Nifty Algo Trader — GCP Deploy"
echo "============================================"
echo ""

cd "$ALGO_DIR"

# ── 1. Pull latest code ──
if [ "$SKIP_PULL" = false ]; then
    echo "[1/5] Pulling latest code..."
    git pull origin main 2>/dev/null || git pull origin master 2>/dev/null || git pull
    echo "  ✓ Code updated"
else
    echo "[1/5] Skipping git pull (--skip-pull)"
fi

# ── 2. Install/update dependencies ──
echo ""
echo "[2/5] Checking Python environment..."
if [ ! -d ".venv" ]; then
    echo "  Creating virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip -q 2>/dev/null
pip install -r requirements.txt -q 2>/dev/null
echo "  ✓ Dependencies ready"

# ── 3. Create directories ──
echo ""
echo "[3/5] Creating directories..."
mkdir -p backtest/paper_trades backtest/results
mkdir -p sapphire/paper_trades sapphire/results
mkdir -p momentum/paper_trades momentum/results
mkdir -p supertrend/paper_trades supertrend/results
echo "  ✓ Directories ready"

# ── 4. Run all backtests (90 days sample data) ──
echo ""
echo "[4/5] Running backtests (90 days each)..."
echo ""

echo "  [4a] EMA Crossover..."
cd "$ALGO_DIR/backtest"
$PYTHON main.py --days 90 --no-charts 2>&1 | tail -5
echo ""

echo "  [4b] Sapphire Strangle..."
cd "$ALGO_DIR/sapphire"
$PYTHON main.py --days 90 --no-charts 2>&1 | tail -5
echo ""

echo "  [4c] Momentum..."
cd "$ALGO_DIR/momentum"
$PYTHON main.py --days 90 --no-charts 2>&1 | tail -5
echo ""

echo "  [4d] Supertrend VWAP..."
cd "$ALGO_DIR/supertrend"
$PYTHON main.py --days 90 --no-charts 2>&1 | tail -5
echo ""

echo "  ✓ All backtests complete"

# ── 5. Regenerate dashboard & tradebook HTML ──
echo ""
echo "[5/5] Generating dashboard & tradebook..."
cd "$ALGO_DIR"

# Load .env if exists (for email config)
if [ -f ".env" ]; then
    export $(grep -v '^\#' .env | xargs) 2>/dev/null
fi

$PYTHON daily_report.py --no-email
echo "  ✓ dashboard.html generated"
echo "  ✓ tradebook.html generated"

# ── Summary ──
echo ""
echo "============================================"
echo "  Deploy complete!"
echo "============================================"
echo ""

# Count trades
TOTAL=$($PYTHON -c "
import pandas as pd
t = 0
for f in ['backtest/paper_trades/paper_trade_log.csv',
           'sapphire/paper_trades/sapphire_trade_log.csv',
           'momentum/paper_trades/momentum_trade_log.csv',
           'supertrend/paper_trades/supertrend_trade_log.csv']:
    try:
        t += len(pd.read_csv(f))
    except: pass
print(t)
" 2>/dev/null || echo "0")

echo "  Total trades in tradebook: $TOTAL"
echo ""

# Start web server if not running
if ! pgrep -f "http.server 8080" > /dev/null; then
    nohup python3 -m http.server 8080 > /dev/null 2>&1 &
    echo "  Dashboard server started on port 8080"
else
    echo "  Dashboard server already running on port 8080"
fi

PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "YOUR_VM_IP")
echo ""
echo "  Dashboard:  http://$PUBLIC_IP:8080/dashboard.html"
echo "  Tradebook:  http://$PUBLIC_IP:8080/tradebook.html"
echo ""
