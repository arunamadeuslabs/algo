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

# Build cron command that loads .env before running
# Mon-Fri at 09:10 IST (timezone set in cloud_setup.sh)
CRON_CMD="10 9 * * 1-5 cd $ALGO_DIR && export \$(grep -v '^\#' $ALGO_DIR/.env | xargs) && $PYTHON $LAUNCHER >> $LOGFILE 2>&1"

# Add to crontab (replacing any existing algo entry)
(crontab -l 2>/dev/null | grep -v "launcher.py" ; echo "$CRON_CMD") | crontab -

echo ""
echo "============================================"
echo "  Cron job installed!"
echo "============================================"
echo ""
echo "  Schedule:  Mon-Fri at 09:10 AM IST"
echo "  Command:   $PYTHON $LAUNCHER"
echo "  Log:       $LOGFILE"
echo ""
echo "  Current crontab:"
crontab -l | grep launcher
echo ""
echo "  To remove: crontab -e  (delete the launcher line)"
echo "  To verify: crontab -l"
echo ""
