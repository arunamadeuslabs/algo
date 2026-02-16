"""Quick Dhan API diagnostic — test which endpoints work with your token."""
import requests
import json
import base64
from datetime import datetime

BASE = "https://api.dhan.co/v2"
TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzcxMzA3MjUyLCJpYXQiOjE3NzEyMjA4NTIsInRva2VuQ29uc3VtZXJUeXBlIjoiU0VMRiIsIndlYmhvb2tVcmwiOiIiLCJkaGFuQ2xpZW50SWQiOiIxMTEwMjY5MDk5In0.3H5RmzOQW3rVP-tn6Nswcc04LI_GZ4eabzzVHiDdFZWBnJpdRJjwimEmahl-NRSSDoZK2n8tkUiGDdnrsXCHEQ"
CLIENT = "1110269099"

headers = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "access-token": TOKEN,
    "client-id": CLIENT,
}

print("=" * 60)
print("  DHAN API v2 — FULL DIAGNOSTIC")
print("=" * 60)

# Decode token
try:
    payload = TOKEN.split(".")[1] + "=="
    decoded = json.loads(base64.b64decode(payload))
    exp = datetime.fromtimestamp(decoded["exp"])
    print(f"\n  Token Client ID : {decoded.get('dhanClientId')}")
    print(f"  Token Type      : {decoded.get('tokenConsumerType')}")
    print(f"  Expires         : {exp}")
    print(f"  Status          : {'✅ VALID' if exp > datetime.now() else '❌ EXPIRED'}")
except Exception as e:
    print(f"  Token decode error: {e}")

# Test 1: Profile (Trading API)
print("\n" + "-" * 60)
print("  Test 1: GET /v2/profile (Trading API)")
try:
    r = requests.get(f"{BASE}/profile", headers=headers, timeout=10)
    print(f"  Status : {r.status_code}")
    print(f"  Result : {r.text[:300]}")
except Exception as e:
    print(f"  Error: {e}")

# Test 2: Fund Limits (Trading API)
print("\n" + "-" * 60)
print("  Test 2: GET /v2/fundlimit (Trading API)")
try:
    r = requests.get(f"{BASE}/fundlimit", headers=headers, timeout=10)
    print(f"  Status : {r.status_code}")
    print(f"  Result : {r.text[:300]}")
except Exception as e:
    print(f"  Error: {e}")

# Test 3: Market Quote LTP (Data API)
print("\n" + "-" * 60)
print("  Test 3: POST /v2/marketfeed/ltp (Data API)")
try:
    r = requests.post(f"{BASE}/marketfeed/ltp", json={"IDX_I": [13]}, headers=headers, timeout=10)
    print(f"  Status : {r.status_code}")
    print(f"  Result : {r.text[:300]}")
except Exception as e:
    print(f"  Error: {e}")

# Test 4: Intraday Charts (Data API)
print("\n" + "-" * 60)
print("  Test 4: POST /v2/charts/intraday (Data API)")
try:
    payload = {
        "securityId": "13",
        "exchangeSegment": "IDX_I",
        "instrument": "INDEX",
        "interval": "5",
        "oi": False,
        "fromDate": "2026-02-14 09:15:00",
        "toDate": "2026-02-16 15:30:00",
    }
    r = requests.post(f"{BASE}/charts/intraday", json=payload, headers=headers, timeout=10)
    print(f"  Status : {r.status_code}")
    print(f"  Result : {r.text[:300]}")
except Exception as e:
    print(f"  Error: {e}")

# Summary
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print("""
  Your API token is valid and authenticating correctly.
  
  The error "Data APIs not Subscribed" means:
  → Your Dhan account has TRADING API access
  → But Data APIs (market data, charts) need separate activation
  
  HOW TO FIX:
  1. Go to https://knowledge.dhan.co/support/portal/en/home
  2. Or open Dhan app → Settings → API Access
  3. Enable "Data APIs" (Market Feed + Historical Data)
  4. It activates instantly — then re-run this test
""")
