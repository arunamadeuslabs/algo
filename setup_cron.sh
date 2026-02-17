#!/bin/bash
# ============================================================
#  Setup cron job for daily algo trading (Linux)
#  Equivalent of setup_scheduler.ps1 for Windows
# ============================================================

ALGO_DIR="$HOME/algo"
PYTHON="$ALGO_DIR/.venv/bin/python3"
LAUNCHER="$ALGO_DIR/launcher.py"
LOGFILE="$ALGO_DIR/launcher.log"

echo "Setting up cron job for Nifty Algo Trader..."

# Verify files exist
if [ ! -f "$LAUNCHER" ]; then
    echo "ERROR: launcher.py not found at $LAUNCHER"
    exit 1
fi

if [ ! -f "$PYTHON" ]; then
    echo "ERROR: Python venv not found. Run cloud_setup.sh first."
    exit 1
fi

if [ ! -f "$ALGO_DIR/.env" ]; then
    echo "WARNING: .env file not found. Create it with your Dhan credentials."
fi

REPORT_SCRIPT="$ALGO_DIR/daily_report.py"
REPORT_LOG="$ALGO_DIR/daily_report.log"

# Build cron commands that load .env before running
# PYTHONIOENCODING=utf-8 prevents charmap codec errors from emoji in print() on GCP
# 1. Mon-Fri at 09:10 IST — Start algos
CRON_LAUNCHER="10 9 * * 1-5 cd $ALGO_DIR && export PYTHONIOENCODING=utf-8 && export \$(grep -v '^\#' $ALGO_DIR/.env | xargs) && $PYTHON $LAUNCHER >> $LOGFILE 2>&1"

# 2. Mon-Fri at 15:40 IST — Send daily report email + update dashboard
CRON_REPORT="40 15 * * 1-5 cd $ALGO_DIR && export PYTHONIOENCODING=utf-8 && export \$(grep -v '^\#' $ALGO_DIR/.env | xargs) && $PYTHON $REPORT_SCRIPT >> $REPORT_LOG 2>&1"

# 3. On reboot — Start dashboard web server
CRON_REBOOT="@reboot cd $ALGO_DIR && python3 -m http.server 8080 > /dev/null 2>&1 &"

# Add to crontab (replacing any existing algo entries)
(crontab -l 2>/dev/null | grep -v "launcher.py" | grep -v "daily_report.py" | grep -v "http.server" ; echo "$CRON_LAUNCHER" ; echo "$CRON_REPORT" ; echo "$CRON_REBOOT") | crontab -

# Start the web server now (if not already running)
if ! pgrep -f "http.server 8080" > /dev/null; then
    cd $ALGO_DIR && nohup python3 -m http.server 8080 > /dev/null 2>&1 &
    echo "  Dashboard web server started on port 8080"
else
    echo "  Dashboard web server already running on port 8080"
fi

# Create directories for new strategies (if missing)
mkdir -p "$ALGO_DIR/momentum/paper_trades" "$ALGO_DIR/momentum/results"
mkdir -p "$ALGO_DIR/supertrend/paper_trades" "$ALGO_DIR/supertrend/results"

# Generate dashboard & tradebook HTML immediately
echo ""
echo "  Generating dashboard & tradebook..."
cd $ALGO_DIR && export $(grep -v '^\#' $ALGO_DIR/.env | xargs) 2>/dev/null
$PYTHON $REPORT_SCRIPT --no-email 2>/dev/null && echo "  ✓ dashboard.html generated" || echo "  ✗ dashboard generation failed"
$PYTHON "$ALGO_DIR/tradebook.py" 2>/dev/null && echo "  ✓ tradebook.html generated" || echo "  ✗ tradebook generation failed"

echo ""
echo "============================================"
echo "  Cron jobs installed!"
echo "============================================"
echo ""
echo "  1. Algo Launcher:  Mon-Fri at 09:10 AM IST"
echo "     Command: $PYTHON $LAUNCHER"
echo "     Log:     $LOGFILE"
echo ""
echo "  2. Daily Report:   Mon-Fri at 03:40 PM IST"
echo "     Command: $PYTHON $REPORT_SCRIPT"
echo "     Log:     $REPORT_LOG"
echo ""
echo "  3. Dashboard:      Auto-start on reboot"
echo "     URL: http://$(curl -s ifconfig.me):8080/dashboard.html"
echo ""
echo "  Current crontab:"
crontab -l | grep -E "launcher|daily_report|http.server"
echo ""
echo "  To remove: crontab -e  (delete the lines)"
echo "  To verify: crontab -l"
echo ""
