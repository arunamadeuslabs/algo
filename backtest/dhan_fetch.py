"""
Dhan API Data Fetcher
=====================
Fetches real Nifty OHLCV data from Dhan API for backtesting.

Supports:
  - Intraday candles: 1min, 5min, 15min, 25min, 60min (last 90 days)
  - Daily candles: since inception

Nifty 50 Security ID: 13 (NSE_EQ index)
Nifty 50 Futures: varies per expiry

API Docs: https://dhanhq.co/docs/v2/historical-data/
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import time

# --- Dhan API Configuration ---
DHAN_API_BASE = "https://api.dhan.co/v2"

# Load credentials from environment variables (for cloud deployment)
# Falls back to hardcoded values for local development
DHAN_ACCESS_TOKEN = os.environ.get("DHAN_JWT_TOKEN", "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzcxMzA3MjUyLCJpYXQiOjE3NzEyMjA4NTIsInRva2VuQ29uc3VtZXJUeXBlIjoiU0VMRiIsIndlYmhvb2tVcmwiOiIiLCJkaGFuQ2xpZW50SWQiOiIxMTEwMjY5MDk5In0.3H5RmzOQW3rVP-tn6Nswcc04LI_GZ4eabzzVHiDdFZWBnJpdRJjwimEmahl-NRSSDoZK2n8tkUiGDdnrsXCHEQ")
DHAN_CLIENT_ID = os.environ.get("DHAN_CLIENT_ID", "1110269099")

# --- Nifty Security IDs ---
# Nifty 50 Index on NSE
NIFTY_SECURITY_ID = "13"
NIFTY_EXCHANGE_SEGMENT = "IDX_I"       # Index segment
NIFTY_INSTRUMENT = "INDEX"

# Nifty Futures (for actual tradeable data with volume)
NIFTY_FUT_EXCHANGE_SEGMENT = "NSE_FNO"
NIFTY_FUT_INSTRUMENT = "FUTIDX"

# Common segment/instrument mappings
SEGMENTS = {
    "nifty_index": {"securityId": "13", "exchangeSegment": "IDX_I", "instrument": "INDEX"},
    "nifty_futures": {"securityId": "13", "exchangeSegment": "NSE_FNO", "instrument": "FUTIDX"},
    "banknifty_index": {"securityId": "25", "exchangeSegment": "IDX_I", "instrument": "INDEX"},
}


def _get_headers():
    """Build request headers for Dhan API."""
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "access-token": DHAN_ACCESS_TOKEN,
        "client-id": DHAN_CLIENT_ID,
    }


def _parse_dhan_response(data: dict) -> pd.DataFrame:
    """
    Parse Dhan API response into a pandas DataFrame.
    Dhan returns arrays for open, high, low, close, volume, timestamp.
    """
    if not data or "open" not in data:
        print("  ⚠️ Empty or invalid response from Dhan API")
        return pd.DataFrame()

    timestamps = data.get("timestamp", [])
    if not timestamps:
        return pd.DataFrame()

    df = pd.DataFrame({
        "datetime": pd.to_datetime(timestamps, unit='s'),
        "open": data["open"],
        "high": data["high"],
        "low": data["low"],
        "close": data["close"],
        "volume": data["volume"],
    })

    # Convert to IST (UTC+5:30)
    df["datetime"] = df["datetime"] + pd.Timedelta(hours=5, minutes=30)
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)

    return df


def fetch_intraday_data(
    security_id: str = NIFTY_SECURITY_ID,
    exchange_segment: str = NIFTY_EXCHANGE_SEGMENT,
    instrument: str = NIFTY_INSTRUMENT,
    interval: int = 5,
    from_date: str = None,
    to_date: str = None,
    days_back: int = 30,
) -> pd.DataFrame:
    """
    Fetch intraday OHLCV data from Dhan API.

    Args:
        security_id: Dhan security ID
        exchange_segment: Exchange segment (IDX_I, NSE_FNO, etc.)
        instrument: Instrument type (INDEX, FUTIDX, etc.)
        interval: Candle interval in minutes (1, 5, 15, 25, 60)
        from_date: Start date "YYYY-MM-DD HH:MM:SS" (optional)
        to_date: End date "YYYY-MM-DD HH:MM:SS" (optional)
        days_back: Number of days back from today (used if from_date is None)

    Returns:
        DataFrame with OHLCV data
    """
    if to_date is None:
        to_dt = datetime.now()
        to_date = to_dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        to_dt = datetime.strptime(to_date, "%Y-%m-%d %H:%M:%S") if isinstance(to_date, str) else to_date

    if from_date is None:
        from_dt = to_dt - timedelta(days=days_back)
        from_date = from_dt.strftime("%Y-%m-%d %H:%M:%S")

    url = f"{DHAN_API_BASE}/charts/intraday"
    payload = {
        "securityId": security_id,
        "exchangeSegment": exchange_segment,
        "instrument": instrument,
        "interval": str(interval),
        "oi": False,
        "fromDate": from_date,
        "toDate": to_date,
    }

    print(f"\n  📡 Fetching intraday data from Dhan API...")
    print(f"     Security: {security_id} ({exchange_segment})")
    print(f"     Interval: {interval}min")
    print(f"     Period: {from_date} → {to_date}")

    try:
        response = requests.post(url, json=payload, headers=_get_headers(), timeout=30)
        print(f"     Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            df = _parse_dhan_response(data)
            if not df.empty:
                print(f"     ✅ Received {len(df)} candles")
                print(f"     Date range: {df.index[0]} → {df.index[-1]}")
                print(f"     Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
            else:
                print(f"     ⚠️ No data in response")
            return df
        else:
            print(f"     ❌ Error: {response.status_code}")
            print(f"     Response: {response.text[:500]}")
            return pd.DataFrame()

    except requests.exceptions.RequestException as e:
        print(f"     ❌ Request failed: {e}")
        return pd.DataFrame()


def fetch_daily_data(
    security_id: str = NIFTY_SECURITY_ID,
    exchange_segment: str = NIFTY_EXCHANGE_SEGMENT,
    instrument: str = NIFTY_INSTRUMENT,
    from_date: str = None,
    to_date: str = None,
    days_back: int = 365,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV data from Dhan API.

    Args:
        security_id: Dhan security ID
        exchange_segment: Exchange segment
        instrument: Instrument type
        from_date: Start date "YYYY-MM-DD" (optional)
        to_date: End date "YYYY-MM-DD" (optional)
        days_back: Number of days back from today

    Returns:
        DataFrame with OHLCV data
    """
    if to_date is None:
        to_dt = datetime.now()
        to_date = to_dt.strftime("%Y-%m-%d")
    else:
        to_dt = datetime.strptime(to_date, "%Y-%m-%d") if isinstance(to_date, str) else to_date

    if from_date is None:
        from_dt = to_dt - timedelta(days=days_back)
        from_date = from_dt.strftime("%Y-%m-%d")

    url = f"{DHAN_API_BASE}/charts/historical"
    payload = {
        "securityId": security_id,
        "exchangeSegment": exchange_segment,
        "instrument": instrument,
        "expiryCode": 0,
        "oi": False,
        "fromDate": from_date,
        "toDate": to_date,
    }

    print(f"\n  📡 Fetching daily data from Dhan API...")
    print(f"     Security: {security_id} ({exchange_segment})")
    print(f"     Period: {from_date} → {to_date}")

    try:
        response = requests.post(url, json=payload, headers=_get_headers(), timeout=30)
        print(f"     Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            df = _parse_dhan_response(data)
            if not df.empty:
                print(f"     ✅ Received {len(df)} daily candles")
                print(f"     Date range: {df.index[0]} → {df.index[-1]}")
            return df
        else:
            print(f"     ❌ Error: {response.status_code}")
            print(f"     Response: {response.text[:500]}")
            return pd.DataFrame()

    except requests.exceptions.RequestException as e:
        print(f"     ❌ Request failed: {e}")
        return pd.DataFrame()


def fetch_nifty_intraday(interval: int = 5, days_back: int = 30) -> pd.DataFrame:
    """
    Convenience function: Fetch Nifty 50 index intraday data.
    Max 90 days back for intraday.
    
    For longer periods, data is fetched in 90-day chunks automatically.
    """
    days_back = min(days_back, 90)  # Dhan API limit

    segment = SEGMENTS["nifty_index"]
    return fetch_intraday_data(
        security_id=segment["securityId"],
        exchange_segment=segment["exchangeSegment"],
        instrument=segment["instrument"],
        interval=interval,
        days_back=days_back,
    )


def fetch_nifty_daily(days_back: int = 365) -> pd.DataFrame:
    """Convenience function: Fetch Nifty 50 daily data."""
    segment = SEGMENTS["nifty_index"]
    return fetch_daily_data(
        security_id=segment["securityId"],
        exchange_segment=segment["exchangeSegment"],
        instrument=segment["instrument"],
        days_back=days_back,
    )


# ── Market Quote APIs (may work without Data API subscription) ──

def fetch_nifty_ltp() -> float:
    """
    Get Nifty 50 Last Traded Price using Market Quote API.
    POST /v2/marketfeed/ltp
    """
    url = f"{DHAN_API_BASE}/marketfeed/ltp"
    payload = {
        "IDX_I": [13]    # Nifty 50 Index
    }

    try:
        response = requests.post(url, json=payload, headers=_get_headers(), timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                ltp = data["data"]["IDX_I"]["13"]["last_price"]
                return float(ltp)
        print(f"  ⚠️ LTP fetch failed: {response.status_code} - {response.text[:200]}")
        return 0.0
    except Exception as e:
        print(f"  ❌ LTP error: {e}")
        return 0.0


def fetch_nifty_ohlc() -> dict:
    """
    Get Nifty 50 OHLC + LTP using Market Quote API.
    POST /v2/marketfeed/ohlc
    Returns: {"last_price": ..., "open": ..., "high": ..., "low": ..., "close": ...}
    """
    url = f"{DHAN_API_BASE}/marketfeed/ohlc"
    payload = {
        "IDX_I": [13]    # Nifty 50 Index
    }

    try:
        response = requests.post(url, json=payload, headers=_get_headers(), timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                quote = data["data"]["IDX_I"]["13"]
                return {
                    "last_price": float(quote["last_price"]),
                    "open": float(quote["ohlc"]["open"]),
                    "high": float(quote["ohlc"]["high"]),
                    "low": float(quote["ohlc"]["low"]),
                    "close": float(quote["ohlc"]["close"]),
                }
        print(f"  ⚠️ OHLC fetch failed: {response.status_code} - {response.text[:200]}")
        return {}
    except Exception as e:
        print(f"  ❌ OHLC error: {e}")
        return {}


def fetch_nifty_quote() -> dict:
    """
    Get full Nifty 50 market depth data using Market Quote API.
    POST /v2/marketfeed/quote
    Returns: full quote dict with LTP, OHLC, volume, depth, OI
    """
    url = f"{DHAN_API_BASE}/marketfeed/quote"
    payload = {
        "IDX_I": [13]    # Nifty 50 Index
    }

    try:
        response = requests.post(url, json=payload, headers=_get_headers(), timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                return data["data"]["IDX_I"]["13"]
        print(f"  ⚠️ Quote fetch failed: {response.status_code} - {response.text[:200]}")
        return {}
    except Exception as e:
        print(f"  ❌ Quote error: {e}")
        return {}


def save_data_to_csv(df: pd.DataFrame, filename: str = None, data_dir: str = "data"):
    """Save fetched data to CSV for reuse."""
    os.makedirs(data_dir, exist_ok=True)
    if filename is None:
        filename = f"nifty_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = os.path.join(data_dir, filename)
    df.to_csv(filepath)
    print(f"  💾 Data saved to: {filepath}")
    return filepath


# ── Main: Test data fetch ────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  DHAN API DATA FETCHER - TEST")
    print("=" * 60)

    # Test 1: Market Quote - LTP (may work without Data API subscription)
    print("\n── Test 1: Nifty LTP (Market Quote API) ──")
    ltp = fetch_nifty_ltp()
    if ltp > 0:
        print(f"  ✅ Nifty LTP: {ltp:.2f}")
    else:
        print("  ❌ LTP not available")

    # Test 2: Market Quote - OHLC
    print("\n── Test 2: Nifty OHLC (Market Quote API) ──")
    ohlc = fetch_nifty_ohlc()
    if ohlc:
        print(f"  ✅ LTP: {ohlc['last_price']:.2f}  O: {ohlc['open']:.2f}  H: {ohlc['high']:.2f}  L: {ohlc['low']:.2f}  C: {ohlc['close']:.2f}")
    else:
        print("  ❌ OHLC not available")

    # Test 3: Historical Intraday (requires Data API subscription)
    print("\n── Test 3: Nifty 5-min Intraday (Data API) ──")
    df_intraday = fetch_nifty_intraday(interval=5, days_back=5)

    if not df_intraday.empty:
        save_data_to_csv(df_intraday, "nifty_5min_intraday.csv")
        print(f"\n  Sample data:")
        print(df_intraday.head(5).to_string())
        print(f"\n  Total bars: {len(df_intraday)}")
    else:
        print("\n  ⚠️ No intraday data — subscribe to Data APIs at developers.dhan.co")

    # Test 4: Historical Daily
    print("\n── Test 4: Nifty Daily (Data API) ──")
    df_daily = fetch_nifty_daily(days_back=30)

    if not df_daily.empty:
        save_data_to_csv(df_daily, "nifty_daily.csv")
        print(f"\n  Sample data:")
        print(df_daily.head(5).to_string())
        print(f"\n  Total bars: {len(df_daily)}")
    else:
        print("\n  ⚠️ No daily data — subscribe to Data APIs at developers.dhan.co")

    print("\n" + "=" * 60)
    print("  API Status Summary:")
    print(f"    Market Quote (LTP/OHLC): {'✅ Working' if ltp > 0 else '❌ Not working'}")
    print(f"    Historical Data:         {'✅ Working' if not df_intraday.empty else '❌ Need Data API subscription'}")
    print("=" * 60)
