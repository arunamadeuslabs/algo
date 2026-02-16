#!/bin/bash
# ============================================================
#  Nifty Algo Trader — Google Cloud VM Setup Script
#  Run this ONCE after SSH-ing into your new VM
# ============================================================

set -e
echo "============================================"
echo "  Setting up Nifty Algo Trader on GCP"
echo "============================================"

# 1. System updates
echo "[1/6] Updating system..."
sudo apt update && sudo apt upgrade -y

# 2. Install Python 3.11+
echo "[2/6] Installing Python..."
sudo apt install -y python3 python3-pip python3-venv git

# 3. Set timezone to IST (critical for market hours!)
echo "[3/6] Setting timezone to IST..."
sudo timedatectl set-timezone Asia/Kolkata
echo "  Timezone: $(timedatectl | grep 'Time zone')"

# 4. Clone repo (replace with your repo URL)
echo "[4/6] Cloning repository..."
cd ~
if [ ! -d "algo" ]; then
    echo "  ⚠️  Clone your repo first:"
    echo "  git clone https://github.com/YOUR_USERNAME/algo.git"
    echo "  Then re-run this script."
    exit 1
fi
cd algo

# 5. Create virtual environment & install deps
echo "[5/6] Setting up Python environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 6. Create .env file (you'll fill in your credentials)
echo "[6/6] Creating .env file..."
if [ ! -f ".env" ]; then
    cat > .env << 'EOF'
# Dhan API Credentials
DHAN_CLIENT_ID=1110269099
DHAN_JWT_TOKEN=PASTE_YOUR_NEW_TOKEN_HERE
EOF
    echo "  ⚠️  IMPORTANT: Edit .env and paste your Dhan JWT token!"
    echo "  Run: nano .env"
else
    echo "  .env already exists, skipping."
fi

# Create paper_trades directories
mkdir -p backtest/paper_trades
mkdir -p sapphire/paper_trades
mkdir -p momentum/paper_trades
mkdir -p backtest/results
mkdir -p sapphire/results
mkdir -p momentum/results

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "  Next steps:"
echo "  1. Edit .env with your Dhan JWT token: nano .env"
echo "  2. Set up cron + dashboard: bash setup_cron.sh"
echo "  3. Test manually: python3 launcher.py --status"
echo "  4. Open dashboard: http://$(curl -s ifconfig.me):8080/dashboard.html"
echo ""
