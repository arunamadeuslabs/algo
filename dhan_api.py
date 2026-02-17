"""
Dhan API Client — Shared Module
================================
Single source of truth for all Dhan API data fetching.
Credentials loaded from .env — no hardcoded tokens.

Usage:
    from dhan_api import get_data, fetch_ltp

    df = get_data("nifty", days=90)        # 90 days of 5-min Nifty data
    df = get_data("banknifty", days=90)    # 90 days of 5-min Bank Nifty data
    ltp = fetch_ltp("nifty")               # Current Nifty LTP
"""

import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Fix encoding for Windows/GCP
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Load credentials from .env ──
_ROOT = Path(__file__).parent
try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass  # env vars must be set manually

# ── API Configuration (from environment only) ──
API_BASE = "https://api.dhan.co/v2"
ACCESS_TOKEN = os.environ.get("DHAN_JWT_TOKEN", "")
CLIENT_ID = os.environ.get("DHAN_CLIENT_ID", "")

if not ACCESS_TOKEN:
    print("  ⚠️  DHAN_JWT_TOKEN not set — add it to .env file or set as env var")

# ── Security Definitions ──
SECURITIES = {
    "nifty": {
        "id": "13",
        "segment": "IDX_I",
        "instrument": "INDEX",
        "name": "Nifty 50",
    },
    "banknifty": {
        "id": "25",
        "segment": "IDX_I",
        "instrument": "INDEX",
        "name": "Bank Nifty",
    },
    "finnifty": {
        "id": "27",
        "segment": "IDX_I",
        "instrument": "INDEX",
        "name": "Fin Nifty",
    },
    "midcapnifty": {
        "id": "442",
        "segment": "IDX_I",
        "instrument": "INDEX",
        "name": "Midcap Nifty",
    },
    "sensex": {
        "id": "51",
        "segment": "IDX_I",
        "instrument": "INDEX",
        "name": "Sensex",
    },
}

# ── Interval mapping ──
TIMEFRAME_MAP = {"1min": 1, "5min": 5, "15min": 15, "1H": 60}


# ── Internal helpers ──

def _headers():
    """Build API request headers."""
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "access-token": ACCESS_TOKEN,
        "client-id": CLIENT_ID,
    }


def _parse_response(data: dict) -> pd.DataFrame:
    """Parse Dhan OHLCV response into DataFrame."""
    if not data or "open" not in data:
        return pd.DataFrame()
    timestamps = data.get("timestamp", [])
    if not timestamps:
        return pd.DataFrame()
    df = pd.DataFrame({
        "datetime": pd.to_datetime(timestamps, unit="s"),
        "open": data["open"],
        "high": data["high"],
        "low": data["low"],
        "close": data["close"],
        "volume": data["volume"],
    })
    df["datetime"] = df["datetime"] + pd.Timedelta(hours=5, minutes=30)  # UTC → IST
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)
    return df


def _filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only market hours: 9:15 - 15:30 IST."""
    if df.empty:
        return df
    df = df[(df.index.hour >= 9) & (df.index.hour <= 15)]
    df = df[~((df.index.hour == 9) & (df.index.minute < 15))]
    df = df[~((df.index.hour == 15) & (df.index.minute > 30))]
    return df


# ── API Fetch Functions ──

def _fetch_intraday_chunk(sec: dict, interval: int,
                          from_date: str, to_date: str) -> pd.DataFrame:
    """Fetch a single chunk of intraday data (max 90 days per call)."""
    url = f"{API_BASE}/charts/intraday"
    payload = {
        "securityId": sec["id"],
        "exchangeSegment": sec["segment"],
        "instrument": sec["instrument"],
        "interval": str(interval),
        "oi": False,
        "fromDate": from_date,
        "toDate": to_date,
    }
    try:
        resp = requests.post(url, json=payload, headers=_headers(), timeout=30)
        if resp.status_code == 200:
            return _parse_response(resp.json())
        print(f"     API error {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"     Request failed: {e}")
    return pd.DataFrame()


def fetch_intraday(symbol: str, interval: int = 5,
                   days_back: int = 90) -> pd.DataFrame:
    """
    Fetch intraday data in 85-day chunks with automatic stitching.
    Handles Dhan's 90-day intraday limit transparently.
    """
    sec = SECURITIES.get(symbol)
    if not sec:
        raise ValueError(f"Unknown symbol: {symbol}. Use: {list(SECURITIES.keys())}")
    if not ACCESS_TOKEN:
        print("  ❌ DHAN_JWT_TOKEN not set — cannot fetch data")
        return pd.DataFrame()

    chunk_size = 85
    all_dfs = []
    end_dt = datetime.now()
    remaining = days_back
    chunk_num = 0

    print(f"\n  📡 Fetching {days_back} days of {sec['name']} {interval}min data...")

    while remaining > 0:
        chunk_days = min(remaining, chunk_size)
        chunk_end = end_dt - timedelta(days=(days_back - remaining))
        chunk_start = chunk_end - timedelta(days=chunk_days)
        chunk_num += 1

        df = _fetch_intraday_chunk(
            sec, interval,
            from_date=chunk_start.strftime("%Y-%m-%d %H:%M:%S"),
            to_date=chunk_end.strftime("%Y-%m-%d %H:%M:%S"),
        )
        if not df.empty:
            all_dfs.append(df)
            print(f"     Chunk {chunk_num}: {len(df)} bars "
                  f"({df.index[0].date()} → {df.index[-1].date()})")

        remaining -= chunk_days
        if remaining > 0:
            time.sleep(2)  # Respect rate limits

    if not all_dfs:
        print("  ⚠️ No data returned from Dhan API")
        return pd.DataFrame()

    combined = pd.concat(all_dfs)
    combined = combined[~combined.index.duplicated(keep="first")]
    combined.sort_index(inplace=True)
    combined = _filter_market_hours(combined)

    days = combined.index.normalize().nunique()
    print(f"  ✅ {sec['name']}: {len(combined)} bars, {days} trading days")
    print(f"     {combined.index[0].date()} → {combined.index[-1].date()}")
    print(f"     ₹{combined['close'].min():.0f} — ₹{combined['close'].max():.0f}")

    return combined


def fetch_daily(symbol: str, days_back: int = 90) -> pd.DataFrame:
    """Fetch daily OHLCV data."""
    sec = SECURITIES.get(symbol)
    if not sec:
        raise ValueError(f"Unknown symbol: {symbol}")
    if not ACCESS_TOKEN:
        return pd.DataFrame()

    url = f"{API_BASE}/charts/historical"
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days_back)

    payload = {
        "securityId": sec["id"],
        "exchangeSegment": sec["segment"],
        "instrument": sec["instrument"],
        "expiryCode": 0,
        "oi": False,
        "fromDate": start_dt.strftime("%Y-%m-%d"),
        "toDate": end_dt.strftime("%Y-%m-%d"),
    }

    try:
        resp = requests.post(url, json=payload, headers=_headers(), timeout=30)
        if resp.status_code == 200:
            df = _parse_response(resp.json())
            if not df.empty:
                print(f"  ✅ {sec['name']} daily: {len(df)} candles")
            return df
        print(f"  ❌ Daily API error: {resp.status_code}")
    except Exception as e:
        print(f"  ❌ Daily fetch error: {e}")
    return pd.DataFrame()


def fetch_ltp(symbol: str) -> float:
    """Get last traded price."""
    sec = SECURITIES.get(symbol)
    if not sec or not ACCESS_TOKEN:
        return 0.0

    url = f"{API_BASE}/marketfeed/ltp"
    payload = {sec["segment"]: [int(sec["id"])]}

    try:
        resp = requests.post(url, json=payload, headers=_headers(), timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "success":
                return float(data["data"][sec["segment"]][sec["id"]]["last_price"])
    except Exception:
        pass
    return 0.0


# ── Main Entry Point: Smart Fetch with Caching ──

def get_data(symbol: str, days: int = 90, interval: int = 5,
             cache_dir: str = None) -> pd.DataFrame:
    """
    Fetch data from Dhan API with smart caching.

    - Reuses cached CSV if data is fresh (last bar from today/yesterday)
    - Fetches from API if cache is stale or missing
    - Saves fetched data to CSV for reuse

    Args:
        symbol: "nifty" or "banknifty"
        days: Number of calendar days to fetch (default: 90)
        interval: Candle interval in minutes (default: 5)
        cache_dir: Directory for cache files (default: <project>/data/)

    Returns:
        DataFrame with OHLCV data indexed by datetime
    """
    if cache_dir is None:
        cache_dir = str(_ROOT / "data")
    os.makedirs(cache_dir, exist_ok=True)

    cache_file = os.path.join(cache_dir, f"{symbol}_{interval}min_{days}d.csv")

    # Check cache freshness
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, parse_dates=["datetime"],
                             index_col="datetime")
            df.sort_index(inplace=True)
            last_bar = df.index[-1]
            days_old = (datetime.now() - last_bar).days

            if days_old <= 1:
                trading_days = df.index.normalize().nunique()
                print(f"  ✅ Cached {SECURITIES[symbol]['name']}: "
                      f"{len(df)} bars, {trading_days} trading days")
                print(f"     {df.index[0].date()} → {df.index[-1].date()}")
                return df
            else:
                print(f"  🔄 Cache is {days_old} days old — refreshing from API...")
        except Exception:
            pass

    # Fetch fresh data from API
    df = fetch_intraday(symbol, interval=interval, days_back=days)

    if not df.empty:
        df.to_csv(cache_file)
        print(f"  💾 Cached → {cache_file}")

    return df


# ── CLI Test ──

if __name__ == "__main__":
    print("=" * 60)
    print("  DHAN API CLIENT — TEST")
    print("=" * 60)

    for sym in SECURITIES:
        print(f"\n── {SECURITIES[sym]['name']} ──")
        ltp = fetch_ltp(sym)
        if ltp > 0:
            print(f"  LTP: ₹{ltp:,.2f}")
        else:
            print("  LTP: unavailable")

    print("\n── Nifty 5min (5 days) ──")
    df = get_data("nifty", days=5, interval=5)
    if not df.empty:
        print(f"  Bars: {len(df)}")
        print(df.tail(3).to_string())

    print("\n" + "=" * 60)
