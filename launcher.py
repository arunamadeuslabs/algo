"""
Algo Auto-Launcher
===================
Starts all trading algos (EMA Crossover + Sapphire Strangle + Momentum) automatically.
Designed to be scheduled via Windows Task Scheduler at 09:10 AM Mon-Fri.

Features:
  - Launches all algos as background processes
  - Monitors health and restarts if crashed
  - Auto-kills after market close (15:35)
  - Logs everything to launcher.log
  - Sends desktop notification on trade events

Usage:
  python launcher.py                  # Start all algos
  python launcher.py --sapphire       # Sapphire only
  python launcher.py --ema            # EMA crossover only
  python launcher.py --momentum       # Momentum only
  python launcher.py --status         # Check if algos are running
  python launcher.py --stop           # Stop all algos
"""

import subprocess
import sys
import os
import time
import signal
import json
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Load .env file for credentials (cloud deployment)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(os.path.dirname(os.path.abspath(__file__))) / ".env")
except ImportError:
    pass  # python-dotenv not installed, env vars must be set manually

# ── Paths ────────────────────────────────────────────────────
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
BACKTEST_DIR = BASE_DIR / "backtest"
SAPPHIRE_DIR = BASE_DIR / "sapphire"
MOMENTUM_DIR = BASE_DIR / "momentum"
IRONCONDOR_DIR = BASE_DIR / "ironcondor"
PYTHON = sys.executable

PID_FILE = BASE_DIR / ".algo_pids.json"
LOG_FILE = BASE_DIR / "launcher.log"

# ── Algo Definitions ────────────────────────────────────────
ALGOS = {
    "ema": {
        "name": "EMA Crossover Paper Trader",
        "script": str(BACKTEST_DIR / "paper_trading.py"),
        "args": ["--live", "--symbol", "sensex"],
        "cwd": str(BACKTEST_DIR),
        "log": str(BACKTEST_DIR / "paper_trades" / "paper_trading.log"),
    },
    "sapphire": {
        "name": "Sapphire Short Strangle",
        "script": str(SAPPHIRE_DIR / "paper_trading.py"),
        "args": ["--live", "--symbol", "nifty"],
        "cwd": str(SAPPHIRE_DIR),
        "log": str(SAPPHIRE_DIR / "paper_trades" / "sapphire_paper.log"),
    },
    "momentum": {
        "name": "Momentum Dual Confirmation",
        "script": str(MOMENTUM_DIR / "paper_trading.py"),
        "args": ["--live", "--symbol", "nifty"],
        "cwd": str(MOMENTUM_DIR),
        "log": str(MOMENTUM_DIR / "paper_trades" / "momentum_paper.log"),
    },
    "ironcondor": {
        "name": "Iron Condor (Sensex)",
        "script": str(IRONCONDOR_DIR / "paper_trading.py"),
        "args": ["--live", "--symbol", "sensex"],
        "cwd": str(IRONCONDOR_DIR),
        "log": str(IRONCONDOR_DIR / "paper_trades" / "ic_paper.log"),
    },
}

# ── Market Timing ────────────────────────────────────────────
MARKET_OPEN = (9, 10)       # Start algos at 09:10
MARKET_CLOSE = (15, 35)     # Kill algos at 15:35 (after EOD square-off at 15:25)
HEALTH_CHECK_SEC = 120      # Check algo health every 2 minutes
MARKET_DAYS = {0, 1, 2, 3, 4}  # Mon-Fri

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("Launcher")


# ── Process Management ───────────────────────────────────────
def start_algo(algo_key: str) -> subprocess.Popen:
    """Start an algo as a background process."""
    algo = ALGOS[algo_key]
    cmd = [PYTHON, algo["script"]] + algo["args"]

    log.info(f"  Starting {algo['name']}...")
    log.info(f"    CMD: {' '.join(cmd)}")
    log.info(f"    CWD: {algo['cwd']}")

    # Cross-platform: CREATE_NO_WINDOW only exists on Windows
    kwargs = dict(
        cwd=algo["cwd"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    proc = subprocess.Popen(cmd, **kwargs)

    log.info(f"    PID: {proc.pid} — Started")
    return proc


def save_pids(pids: dict):
    """Save running PIDs to file."""
    with open(PID_FILE, "w") as f:
        json.dump(pids, f, indent=2)


def load_pids() -> dict:
    """Load saved PIDs."""
    if PID_FILE.exists():
        try:
            with open(PID_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def is_process_alive(pid: int) -> bool:
    """Check if a process is still running (cross-platform)."""
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x0400, False, pid)  # PROCESS_QUERY_INFORMATION
            if handle:
                exit_code = ctypes.c_ulong()
                kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
                kernel32.CloseHandle(handle)
                return exit_code.value == 259  # STILL_ACTIVE
            return False
        except Exception:
            return False
    else:
        # Linux/Mac
        try:
            os.kill(pid, 0)  # Signal 0 = just check if alive
            return True
        except (ProcessLookupError, PermissionError):
            return False


def kill_process(pid: int):
    """Kill a process by PID."""
    try:
        os.kill(pid, signal.SIGTERM)
        log.info(f"    Killed PID {pid}")
    except (ProcessLookupError, PermissionError, OSError):
        pass


def is_market_day() -> bool:
    """Check if today is a trading day (Mon-Fri)."""
    return datetime.now().weekday() in MARKET_DAYS


def is_market_hours() -> bool:
    """Check if current time is within market hours."""
    now = datetime.now()
    start = now.replace(hour=MARKET_OPEN[0], minute=MARKET_OPEN[1], second=0)
    end = now.replace(hour=MARKET_CLOSE[0], minute=MARKET_CLOSE[1], second=0)
    return start <= now <= end


def desktop_notify(title: str, message: str):
    """Send Windows desktop notification (best-effort)."""
    try:
        from plyer import notification
        notification.notify(title=title, message=message, timeout=10)
    except ImportError:
        pass  # plyer not installed, skip
    except Exception:
        pass


# ── Main Runner ──────────────────────────────────────────────
def run(algo_keys: list):
    """Main loop: start algos, monitor health, stop after market close."""
    log.info("=" * 60)
    log.info("  ALGO AUTO-LAUNCHER")
    log.info("=" * 60)
    log.info(f"  Date: {datetime.now().strftime('%Y-%m-%d %A')}")
    log.info(f"  Algos: {', '.join(algo_keys)}")
    log.info(f"  Python: {PYTHON}")

    if not is_market_day():
        log.info("  Not a trading day (weekend). Exiting.")
        return

    # Wait for market open if too early
    now = datetime.now()
    open_time = now.replace(hour=MARKET_OPEN[0], minute=MARKET_OPEN[1], second=0)
    if now < open_time:
        wait_sec = (open_time - now).total_seconds()
        log.info(f"  Pre-market. Waiting {int(wait_sec // 60)}m until {open_time.strftime('%H:%M')}...")
        time.sleep(wait_sec)

    if now > now.replace(hour=MARKET_CLOSE[0], minute=MARKET_CLOSE[1]):
        log.info("  Market already closed. Exiting.")
        return

    # Start algos
    processes = {}
    pids = {}

    for key in algo_keys:
        proc = start_algo(key)
        processes[key] = proc
        pids[key] = proc.pid

    # Start dashboard server
    dashboard_script = str(BASE_DIR / "dashboard_server.py")
    if os.path.exists(dashboard_script):
        log.info("  Starting dashboard server on port 8080...")
        dash_cmd = [PYTHON, dashboard_script, "--no-browser", "--port", "8080"]
        dash_kwargs = dict(
            cwd=str(BASE_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if sys.platform == "win32":
            dash_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        dash_proc = subprocess.Popen(dash_cmd, **dash_kwargs)
        processes["dashboard"] = dash_proc
        pids["dashboard"] = dash_proc.pid
        log.info(f"    Dashboard PID: {dash_proc.pid} — http://localhost:8080")

    save_pids(pids)
    desktop_notify("Algos Started", f"Running: {', '.join(algo_keys)}")

    log.info(f"\n  All algos started. Monitoring until {MARKET_CLOSE[0]}:{MARKET_CLOSE[1]:02d}...")

    # Monitor loop
    try:
        while True:
            now = datetime.now()
            close_time = now.replace(hour=MARKET_CLOSE[0], minute=MARKET_CLOSE[1], second=0)

            if now >= close_time:
                log.info("\n  Market closed. Shutting down algos...")
                break

            # Health check: restart crashed algos
            for key in algo_keys:
                proc = processes.get(key)
                if proc and proc.poll() is not None:
                    # Process died — restart it
                    exit_code = proc.returncode
                    log.warning(f"  {ALGOS[key]['name']} crashed (exit={exit_code}). Restarting...")
                    desktop_notify("Algo Crashed", f"{ALGOS[key]['name']} restarted")

                    proc = start_algo(key)
                    processes[key] = proc
                    pids[key] = proc.pid
                    save_pids(pids)

            time.sleep(HEALTH_CHECK_SEC)

    except KeyboardInterrupt:
        log.info("\n  Manual interrupt received.")

    # Shutdown all
    for key in algo_keys:
        proc = processes.get(key)
        if proc and proc.poll() is None:
            log.info(f"  Stopping {ALGOS[key]['name']}...")
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
            log.info(f"    Stopped (PID {proc.pid})")

    # Cleanup PID file
    if PID_FILE.exists():
        PID_FILE.unlink()

    desktop_notify("Algos Stopped", "All algos shut down for the day")

    # Generate daily report & send email
    log.info("  Generating daily report...")
    try:
        report_script = str(BASE_DIR / "daily_report.py")
        result = subprocess.run(
            [PYTHON, report_script],
            cwd=str(BASE_DIR),
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            log.info("  Daily report sent successfully.")
        else:
            log.warning(f"  Daily report failed: {result.stderr[:200]}")
    except Exception as e:
        log.warning(f"  Could not generate daily report: {e}")

    log.info("\n  All algos stopped. Session complete.")
    log.info("=" * 60)


def check_status():
    """Check if algos are currently running."""
    pids = load_pids()
    if not pids:
        print("  No algos are tracked. PID file not found.")
        return

    print(f"\n  {'Algo':<30} {'PID':<10} {'Status'}")
    print(f"  {'─' * 55}")
    for key, pid in pids.items():
        name = ALGOS.get(key, {}).get("name", key)
        alive = is_process_alive(pid)
        status = "RUNNING" if alive else "STOPPED"
        icon = "●" if alive else "○"
        print(f"  {name:<30} {pid:<10} {icon} {status}")

    # Show latest log line for each
    print()
    for key in pids:
        algo = ALGOS.get(key)
        if algo and os.path.exists(algo["log"]):
            try:
                with open(algo["log"], "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    if lines:
                        last = lines[-1].strip()
                        print(f"  [{key}] Last log: {last}")
            except Exception:
                pass
    print()


def stop_all():
    """Stop all running algos."""
    pids = load_pids()
    if not pids:
        print("  No algos to stop.")
        return

    for key, pid in pids.items():
        name = ALGOS.get(key, {}).get("name", key)
        if is_process_alive(pid):
            kill_process(pid)
            print(f"  Stopped {name} (PID {pid})")
        else:
            print(f"  {name} was not running")

    if PID_FILE.exists():
        PID_FILE.unlink()
    print("  All algos stopped.")


# ── CLI ──────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Algo Auto-Launcher")
    parser.add_argument("--ema", action="store_true", help="Run EMA crossover only")
    parser.add_argument("--sapphire", action="store_true", help="Run Sapphire only")
    parser.add_argument("--momentum", action="store_true", help="Run Momentum only")
    parser.add_argument("--ironcondor", action="store_true", help="Run Iron Condor only")
    parser.add_argument("--status", action="store_true", help="Check running algos")
    parser.add_argument("--stop", action="store_true", help="Stop all algos")

    args = parser.parse_args()

    if args.status:
        check_status()
    elif args.stop:
        stop_all()
    else:
        # Determine which algos to run
        keys = []
        if args.ema:
            keys.append("ema")
        if args.sapphire:
            keys.append("sapphire")
        if args.momentum:
            keys.append("momentum")
        if args.ironcondor:
            keys.append("ironcondor")
        if not keys:
            keys = ["ema", "sapphire", "momentum", "ironcondor"]  # Default: all four

        run(keys)
