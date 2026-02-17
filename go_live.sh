#!/bin/bash
# ============================================================
#  go_live.sh — One-command GCP live paper trading setup
# ============================================================
#  Run this ONCE on your GCP VM to pull latest code, install
#  dependencies, set up cron jobs, start the live dashboard,
#  and enable auto-start on reboot.
#
#  Usage:
#    bash go_live.sh              # Full setup
#    bash go_live.sh --restart    # Just restart dashboard + servers
#    bash go_live.sh --status     # Check what's running
# ============================================================

set -e

ALGO_DIR="$HOME/algo"
PYTHON="$ALGO_DIR/.venv/bin/python3"
LAUNCHER="$ALGO_DIR/launcher.py"
DASHBOARD="$ALGO_DIR/dashboard_server.py"
REPORT="$ALGO_DIR/daily_report.py"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

header() { echo -e "\n${CYAN}[$1]${NC} $2"; }
ok()     { echo -e "  ${GREEN}✓${NC} $1"; }
warn()   { echo -e "  ${YELLOW}⚠${NC} $1"; }
fail()   { echo -e "  ${RED}✗${NC} $1"; }

# ── Status check ─────────────────────────────────────────────
if [ "$1" = "--status" ]; then
    echo ""
    echo "============================================"
    echo "  Algo Paper Trading — Status"
    echo "============================================"
    echo ""

    # Check dashboard
    if pgrep -f "dashboard_server.py" > /dev/null; then
        PID=$(pgrep -f "dashboard_server.py")
        echo -e "  ${GREEN}●${NC} Live Dashboard    PID $PID  (port 8050)"
    else
        echo -e "  ${RED}○${NC} Live Dashboard    NOT RUNNING"
    fi

    # Check static server
    if pgrep -f "http.server 8080" > /dev/null; then
        PID=$(pgrep -f "http.server 8080")
        echo -e "  ${GREEN}●${NC} Static Server     PID $PID  (port 8080)"
    else
        echo -e "  ${RED}○${NC} Static Server     NOT RUNNING"
    fi

    # Check algo processes
    if [ -f "$ALGO_DIR/.algo_pids.json" ]; then
        echo ""
        cd "$ALGO_DIR"
        $PYTHON -c "
import json, os
with open('.algo_pids.json') as f:
    pids = json.load(f)
for name, pid in pids.items():
    try:
        os.kill(pid, 0)
        print(f'  \033[0;32m●\033[0m {name:<18} PID {pid}')
    except:
        print(f'  \033[0;31m○\033[0m {name:<18} PID {pid} (stopped)')
" 2>/dev/null || echo "  No algo PIDs found."
    else
        echo ""
        echo "  No algos running (no PID file)."
    fi

    # Cron jobs
    echo ""
    echo "  Cron jobs:"
    crontab -l 2>/dev/null | grep -E "launcher|daily_report|dashboard" | while read line; do
        echo "    $line"
    done
    echo ""

    PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "YOUR_VM_IP")
    echo "  Live Dashboard: http://$PUBLIC_IP:8050"
    echo "  Tradebook:      http://$PUBLIC_IP:8080/tradebook.html"
    echo ""
    exit 0
fi

# ── Main setup / restart ─────────────────────────────────────
echo ""
echo "============================================"
echo "  Algo Paper Trading — Go Live on GCP"
echo "============================================"
echo "  Date: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo ""

cd "$ALGO_DIR" || { fail "Directory $ALGO_DIR not found. Run cloud_setup.sh first."; exit 1; }

# ── 1. Pull latest code ──────────────────────────────────────
if [ "$1" != "--restart" ]; then
    header "1/7" "Pulling latest code..."
    git pull origin master 2>/dev/null || git pull origin main 2>/dev/null || git pull
    ok "Code updated"
else
    header "1/7" "Skipping git pull (--restart mode)"
fi

# ── 2. Python environment ────────────────────────────────────
header "2/7" "Checking Python environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    ok "Virtual environment created"
fi
source .venv/bin/activate
pip install --upgrade pip -q 2>/dev/null
pip install -r requirements.txt -q 2>/dev/null
ok "Dependencies up to date"

# ── 3. Verify .env ───────────────────────────────────────────
header "3/7" "Checking credentials..."
if [ ! -f ".env" ]; then
    cat > .env << 'EOF'
# Dhan API Credentials
DHAN_CLIENT_ID=1110269099
DHAN_JWT_TOKEN=PASTE_YOUR_TOKEN_HERE
EOF
    warn "Created .env — EDIT IT NOW: nano .env"
    echo ""
    read -p "  Press Enter after editing .env (or Ctrl+C to abort)..."
else
    # Verify token is set
    TOKEN=$(grep DHAN_JWT_TOKEN .env | cut -d= -f2)
    if [ "$TOKEN" = "PASTE_YOUR_TOKEN_HERE" ] || [ -z "$TOKEN" ]; then
        warn "JWT token not set! Edit .env: nano .env"
        read -p "  Press Enter after editing .env (or Ctrl+C to abort)..."
    else
        ok ".env found with JWT token"
    fi
fi

# Load env
export $(grep -v '^\#' .env | xargs) 2>/dev/null

# ── 4. Ensure timezone is IST ────────────────────────────────
header "4/7" "Checking timezone..."
CURRENT_TZ=$(timedatectl show -p Timezone --value 2>/dev/null || echo "unknown")
if [ "$CURRENT_TZ" != "Asia/Kolkata" ]; then
    sudo timedatectl set-timezone Asia/Kolkata
    ok "Timezone set to IST (Asia/Kolkata)"
else
    ok "Already IST"
fi

# ── 5. Create directories ────────────────────────────────────
header "5/7" "Setting up directories..."
mkdir -p backtest/paper_trades backtest/results
mkdir -p sapphire/paper_trades sapphire/results
mkdir -p momentum/paper_trades momentum/results
mkdir -p ironcondor/paper_trades ironcondor/results
ok "All directories ready"

# ── 5b. Seed paper trade data if empty ─────────────────────
# Run backtests to populate trade logs if no data exists yet
if [ ! -f "backtest/paper_trades/paper_trade_log.csv" ] || [ ! -f "ironcondor/paper_trades/ic_trade_log.csv" ]; then
    header "5b" "Seeding initial data via backtests (90 days)..."
    echo "  This runs once to populate the dashboard with historical data."
    echo ""

    cd "$ALGO_DIR/backtest"
    $PYTHON main.py --days 90 --no-charts 2>&1 | tail -3 && ok "EMA backtest done" || warn "EMA backtest failed"

    cd "$ALGO_DIR/sapphire"
    $PYTHON main.py --days 90 --no-charts 2>&1 | tail -3 && ok "Sapphire backtest done" || warn "Sapphire backtest failed"

    cd "$ALGO_DIR/momentum"
    $PYTHON main.py --days 90 --no-charts 2>&1 | tail -3 && ok "Momentum backtest done" || warn "Momentum backtest failed"

    cd "$ALGO_DIR/ironcondor"
    $PYTHON main.py --days 90 --no-charts 2>&1 | tail -3 && ok "Iron Condor backtest done" || warn "Iron Condor backtest failed"

    cd "$ALGO_DIR"
    echo ""
else
    ok "Trade data already exists, skipping backtest seed"
fi

# ── 6. Kill old processes & start fresh ──────────────────────
header "6/7" "Starting services..."

# Kill existing dashboard/static servers
pkill -f "dashboard_server.py" 2>/dev/null && echo "  Stopped old dashboard" || true
pkill -f "http.server 8080" 2>/dev/null && echo "  Stopped old static server" || true
sleep 1

# Start live dashboard on port 8050 (serves everything: dashboard + tradebook + API)
nohup $PYTHON $DASHBOARD --port 8050 --no-browser >> dashboard.log 2>&1 &
DASH_PID=$!
ok "Live dashboard started (PID $DASH_PID, port 8050)"

# Quick test
sleep 2
if curl -s http://localhost:8050/api/data > /dev/null 2>&1; then
    ok "Dashboard API responding"
else
    warn "Dashboard API not responding yet (may need a moment)"
fi

# Generate tradebook & dashboard HTML
$PYTHON daily_report.py --no-email 2>/dev/null && ok "dashboard.html generated" || warn "dashboard.html generation failed"
$PYTHON tradebook.py 2>/dev/null && ok "tradebook.html generated" || warn "tradebook.html generation failed"

# ── 7. Install cron jobs ─────────────────────────────────────
header "7/7" "Setting up cron jobs..."

LOGFILE="$ALGO_DIR/launcher.log"
REPORT_LOG="$ALGO_DIR/daily_report.log"
DASHBOARD_LOG="$ALGO_DIR/dashboard.log"

# Cron entries
CRON_LAUNCHER="10 9 * * 1-5 cd $ALGO_DIR && export PYTHONIOENCODING=utf-8 && export \$(grep -v '^\#' $ALGO_DIR/.env | xargs) && $PYTHON $LAUNCHER >> $LOGFILE 2>&1"
CRON_REPORT="40 15 * * 1-5 cd $ALGO_DIR && export PYTHONIOENCODING=utf-8 && export \$(grep -v '^\#' $ALGO_DIR/.env | xargs) && $PYTHON $REPORT >> $REPORT_LOG 2>&1"
CRON_REBOOT="@reboot sleep 30 && cd $ALGO_DIR && export \$(grep -v '^\#' $ALGO_DIR/.env | xargs) && $PYTHON $DASHBOARD --port 8050 --no-browser >> $DASHBOARD_LOG 2>&1 &"

# Install (replace old entries)
(crontab -l 2>/dev/null | grep -v "launcher.py" | grep -v "daily_report.py" | grep -v "http.server" | grep -v "dashboard_server.py" ; echo "$CRON_LAUNCHER" ; echo "$CRON_REPORT" ; echo "$CRON_REBOOT") | crontab -

ok "Cron: Algos launch Mon-Fri 09:10 AM IST"
ok "Cron: Daily report Mon-Fri 03:40 PM IST"
ok "Cron: Dashboard + static server on reboot"

# ── Done ──────────────────────────────────────────────────────
PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "YOUR_VM_IP")

echo ""
echo "============================================"
echo -e "  ${GREEN}LIVE PAPER TRADING — READY${NC}"
echo "============================================"
echo ""
echo "  Strategies:"
echo "    1. EMA Crossover     (Sensex)  ₹3.0L"
echo "    2. Sapphire Strangle (Nifty)   ₹2.5L"
echo "    3. Momentum          (Nifty)   ₹2.5L"
echo "    4. Iron Condor       (Sensex)  ₹2.0L"
echo "    Total capital: ₹10.0L"
echo ""
echo "  Schedule:"
echo "    09:10 AM  → Algos start (Mon-Fri)"
echo "    03:40 PM  → Daily report emailed"
echo "    24/7      → Dashboard always on"
echo ""
echo "  URLs:"
echo "    Live Dashboard: http://$PUBLIC_IP:8050"
echo "    Tradebook:      http://$PUBLIC_IP:8050/tradebook.html"
echo "    Daily Report:   http://$PUBLIC_IP:8050/dashboard.html"
echo "    API:            http://$PUBLIC_IP:8050/api/data"
echo ""
echo "  Commands:"
echo "    bash go_live.sh --status    # Check status"
echo "    bash go_live.sh --restart   # Restart servers"
echo "    python3 launcher.py --stop  # Stop algos"
echo "    tail -f launcher.log        # Watch algo logs"
echo "    tail -f dashboard.log       # Watch dashboard logs"
echo ""
