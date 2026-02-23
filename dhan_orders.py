"""
Dhan API Order Execution Module
=================================
Places real orders on Dhan using API v2.
Supports: Market/Limit orders, modify, cancel, position tracking.

Usage:
    from dhan_orders import DhanOrderManager

    mgr = DhanOrderManager()
    order_id = mgr.place_order("SELL", "nifty", quantity=25, strike=24500, option_type="CE")
    mgr.cancel_order(order_id)
    positions = mgr.get_positions()

Requires: DHAN_JWT_TOKEN and DHAN_CLIENT_ID in .env

Safety:
    - Double gate: both --real CLI flag AND LIVE_TRADING=true in .env required
    - All orders logged to order_logs/ regardless of mode
    - Paper mode default: without --real, nothing executes

API Docs: https://dhanhq.co/docs/v2/
"""

import os
import sys
import json
import time
import logging
import requests
import pandas as pd
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, List
import io

# Fix encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Load credentials
_ROOT = Path(__file__).parent
try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

API_BASE = "https://api.dhan.co/v2"
ACCESS_TOKEN = os.environ.get("DHAN_JWT_TOKEN", "")
CLIENT_ID = os.environ.get("DHAN_CLIENT_ID", "")

# ── Trading mode gate ──
# Set LIVE_TRADING=true in .env to allow real orders
LIVE_TRADING_ENABLED = os.environ.get("LIVE_TRADING", "false").lower() == "true"

logger = logging.getLogger("DhanOrders")

# ── Exchange Segments ──
EXCHANGE_NSE_FO = "NSE_FNO"
EXCHANGE_BSE_FO = "BSE_FNO"
EXCHANGE_NSE_EQ = "NSE_EQ"

# ── Product Types ──
PRODUCT_INTRADAY = "INTRADAY"   # MIS
PRODUCT_MARGIN = "MARGIN"       # NRML
PRODUCT_CNC = "CNC"             # Delivery

# ── Order Types ──
ORDER_MARKET = "MARKET"
ORDER_LIMIT = "LIMIT"
ORDER_SL = "STOP_LOSS"
ORDER_SLM = "STOP_LOSS_MARKET"

# ── Transaction Types ──
BUY = "BUY"
SELL = "SELL"

# ── Symbol → trading-symbol prefix mapping ──
SYMBOL_MAP = {
    "nifty":       {"prefix": "NIFTY",      "exchange": "NSE_FNO", "step": 50},
    "banknifty":   {"prefix": "BANKNIFTY",  "exchange": "NSE_FNO", "step": 100},
    "finnifty":    {"prefix": "FINNIFTY",   "exchange": "NSE_FNO", "step": 50},
    "midcapnifty": {"prefix": "MIDCPNIFTY", "exchange": "NSE_FNO", "step": 25},
    "sensex":      {"prefix": "SENSEX",     "exchange": "BSE_FNO", "step": 100},
}

# Keep legacy alias
UNDERLYING_IDS = {k: {"security_id": 0, "exchange": v["exchange"], "step": v["step"]}
                  for k, v in SYMBOL_MAP.items()}

# ── Order log file ──
ORDER_LOG_DIR = _ROOT / "order_logs"
ORDER_LOG_DIR.mkdir(exist_ok=True)

# ── Scrip Master cache ──
SCRIP_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
_scrip_cache: Optional[pd.DataFrame] = None
_scrip_cache_date: Optional[date] = None


def _headers() -> dict:
    """Build API request headers."""
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "access-token": ACCESS_TOKEN,
        "client-id": CLIENT_ID,
    }


def _load_scrip_master() -> pd.DataFrame:
    """Download and cache Dhan scrip master CSV (once per day)."""
    global _scrip_cache, _scrip_cache_date
    today = date.today()

    if _scrip_cache is not None and _scrip_cache_date == today:
        return _scrip_cache

    cache_file = ORDER_LOG_DIR / f"scrip_master_{today.isoformat()}.csv"

    if cache_file.exists():
        logger.info("Loading cached scrip master...")
        _scrip_cache = pd.read_csv(cache_file, low_memory=False)
        _scrip_cache_date = today
        return _scrip_cache

    logger.info("Downloading Dhan scrip master (~30 MB)...")
    resp = requests.get(SCRIP_MASTER_URL, timeout=120)
    resp.raise_for_status()

    with open(cache_file, "wb") as f:
        f.write(resp.content)

    _scrip_cache = pd.read_csv(io.StringIO(resp.text), low_memory=False)
    _scrip_cache_date = today
    logger.info(f"Scrip master loaded: {len(_scrip_cache)} instruments")
    return _scrip_cache


class DhanOrderManager:
    """
    Manages real order placement, modification, cancellation via Dhan API v2.
    Includes safety checks, logging, and position tracking.

    Double-gate safety:
        1. Constructor `live=True` must be passed (from --real CLI flag)
        2. LIVE_TRADING=true must be set in .env
        Both must be true for real orders to be placed.
    """

    def __init__(self, live: bool = False):
        """
        Args:
            live: If True AND LIVE_TRADING env is 'true', places real orders.
                  Otherwise logs orders but does not execute.
        """
        self.live = live and LIVE_TRADING_ENABLED
        self.orders: Dict[str, dict] = {}  # order_id -> order details
        self.today_orders: List[dict] = []
        self._order_log_file = ORDER_LOG_DIR / f"orders_{date.today().isoformat()}.json"

        if not ACCESS_TOKEN:
            logger.error("DHAN_JWT_TOKEN not set — cannot place orders")
        if self.live:
            logger.warning("=" * 60)
            logger.warning("  !! LIVE TRADING MODE — REAL ORDERS WILL BE PLACED !!")
            logger.warning("=" * 60)
        else:
            logger.info("  Paper mode — orders will be logged but not executed")

    # ── Security ID Lookup via Scrip Master ─────────────────
    def get_option_security_id(self, symbol: str, strike: float,
                                option_type: str, expiry_date: str = None) -> Optional[str]:
        """
        Look up the Dhan security_id for a specific option contract.
        Uses the Dhan scrip master CSV (downloaded & cached daily).

        Args:
            symbol: "nifty", "banknifty", "sensex", etc.
            strike: Strike price (e.g., 24500)
            option_type: "CE" or "PE"
            expiry_date: "YYYY-MM-DD" (default: nearest weekly expiry)

        Returns:
            Security ID string, or None if not found
        """
        sym_info = SYMBOL_MAP.get(symbol)
        if not sym_info:
            logger.error(f"Unknown symbol: {symbol}")
            return None

        try:
            df = _load_scrip_master()
            prefix = sym_info["prefix"]

            # Filter: trading symbol starts with PREFIX-, correct instrument type
            mask = (
                df["SEM_TRADING_SYMBOL"].str.startswith(f"{prefix}-", na=False) &
                (df["SEM_INSTRUMENT_NAME"] == "OPTIDX") &
                (df["SEM_STRIKE_PRICE"] == float(strike)) &
                (df["SEM_OPTION_TYPE"] == option_type.upper())
            )
            matches = df[mask].copy()

            if matches.empty:
                logger.warning(f"Option not found in scrip master: {symbol} {strike}{option_type}")
                return None

            # Parse expiry dates for sorting
            matches["_expiry"] = pd.to_datetime(matches["SEM_EXPIRY_DATE"], errors="coerce")
            now = pd.Timestamp.now()

            if expiry_date:
                # Match specific expiry
                target = pd.Timestamp(expiry_date)
                matches = matches[matches["_expiry"].dt.date == target.date()]
            else:
                # Nearest future expiry
                matches = matches[matches["_expiry"] >= now]

            if matches.empty:
                logger.warning(f"No valid expiry for {symbol} {strike}{option_type}")
                return None

            matches = matches.sort_values("_expiry")
            sec_id = int(matches.iloc[0]["SEM_SMST_SECURITY_ID"])
            logger.info(f"  Resolved {symbol} {strike}{option_type} -> secId={sec_id}")
            return str(sec_id)

        except Exception as e:
            logger.error(f"Option scrip-master lookup failed: {e}")
            return None

    def get_futures_security_id(self, symbol: str, expiry_date: str = None) -> Optional[str]:
        """Look up the Dhan security_id for a futures contract via scrip master."""
        sym_info = SYMBOL_MAP.get(symbol)
        if not sym_info:
            logger.error(f"Unknown symbol: {symbol}")
            return None

        try:
            df = _load_scrip_master()
            prefix = sym_info["prefix"]

            mask = (
                df["SEM_TRADING_SYMBOL"].str.startswith(f"{prefix}-", na=False) &
                (df["SEM_INSTRUMENT_NAME"] == "FUTIDX")
            )
            matches = df[mask].copy()

            if matches.empty:
                logger.warning(f"Futures not found in scrip master: {symbol}")
                return None

            matches["_expiry"] = pd.to_datetime(matches["SEM_EXPIRY_DATE"], errors="coerce")
            now = pd.Timestamp.now()

            if expiry_date:
                target = pd.Timestamp(expiry_date)
                matches = matches[matches["_expiry"].dt.date == target.date()]
            else:
                matches = matches[matches["_expiry"] >= now]

            if matches.empty:
                logger.warning(f"No valid futures expiry for {symbol}")
                return None

            matches = matches.sort_values("_expiry")
            sec_id = int(matches.iloc[0]["SEM_SMST_SECURITY_ID"])
            logger.info(f"  Resolved {symbol} FUT -> secId={sec_id}")
            return str(sec_id)

        except Exception as e:
            logger.error(f"Futures scrip-master lookup failed: {e}")
            return None

    # ── Order Placement ──────────────────────────────────────
    def place_order(
        self,
        transaction_type: str,     # "BUY" or "SELL"
        symbol: str,               # "nifty", "banknifty", etc.
        quantity: int,             # Total quantity (not lots)
        strike: float = None,     # For options
        option_type: str = None,  # "CE" or "PE" (None = futures)
        order_type: str = ORDER_MARKET,
        price: float = 0,
        trigger_price: float = 0,
        product_type: str = PRODUCT_INTRADAY,
        tag: str = "",
    ) -> Optional[str]:
        """
        Place an order on Dhan.

        Args:
            transaction_type: "BUY" or "SELL"
            symbol: "nifty", "banknifty", "sensex", etc.
            quantity: Total quantity (e.g., 25 for 1 Nifty lot)
            strike: Strike price (for options only)
            option_type: "CE" or "PE" (None for futures)
            order_type: MARKET, LIMIT, STOP_LOSS, STOP_LOSS_MARKET
            price: Limit price (for LIMIT orders)
            trigger_price: Trigger price (for SL orders)
            product_type: INTRADAY (MIS), MARGIN (NRML), CNC
            tag: Correlation tag for tracking

        Returns: order_id string on success, None on failure.
        """
        sym_info = SYMBOL_MAP.get(symbol, {})
        exchange = sym_info.get("exchange", EXCHANGE_NSE_FO)

        # ── PAPER MODE — skip all API calls, return immediately ──
        if not self.live:
            paper_id = f"PAPER_{int(time.time())}_{symbol}"
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "action": "PLACE_ORDER",
                "live": False,
                "symbol": symbol,
                "strike": strike,
                "option_type": option_type,
                "transaction": transaction_type,
                "quantity": quantity,
                "order_type": order_type,
                "price": price,
                "tag": tag,
                "status": "PAPER_MODE",
                "order_id": paper_id,
            }
            self._log_order(log_entry)
            logger.info(f"  [PAPER] {transaction_type} {quantity} {symbol} "
                        f"{strike}{option_type or 'FUT'} @ {order_type}")
            return paper_id

        # ── LIVE MODE — resolve security ID from scrip master ──
        if option_type:
            security_id = self.get_option_security_id(symbol, strike, option_type)
            instrument = "OPTIDX"
        else:
            security_id = self.get_futures_security_id(symbol)
            instrument = "FUTIDX"

        if not security_id:
            logger.error(f"Could not resolve security ID for {symbol} "
                         f"{strike}{option_type or 'FUT'}")
            return None

        order_detail = {
            "transactionType": transaction_type,
            "exchangeSegment": exchange,
            "productType": product_type,
            "orderType": order_type,
            "securityId": security_id,
            "quantity": quantity,
            "price": price if order_type == ORDER_LIMIT else 0,
            "triggerPrice": trigger_price if order_type in (ORDER_SL, ORDER_SLM) else 0,
            "disclosedQuantity": 0,
            "validity": "DAY",
            "afterMarketOrder": False,
            "amoTime": "",
            "boProfitValue": 0,
            "boStopLossValue": 0,
            "drvExpiryDate": "",
            "drvOptionType": option_type if option_type else "",
            "drvStrikePrice": strike if strike else 0,
            "correlationId": tag or f"algo_{datetime.now().strftime('%H%M%S')}",
        }

        # Log the order intent
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": "PLACE_ORDER",
            "live": self.live,
            "symbol": symbol,
            "strike": strike,
            "option_type": option_type,
            "transaction": transaction_type,
            "quantity": quantity,
            "order_type": order_type,
            "price": price,
            "tag": tag,
        }

        # ── LIVE ORDER EXECUTION ──
        url = f"{API_BASE}/orders"
        try:
            logger.info(f"  ⚡ PLACING LIVE ORDER: {transaction_type} {quantity} "
                        f"{symbol} {strike}{option_type or 'FUT'} @ {order_type}")

            resp = requests.post(url, json=order_detail, headers=_headers(), timeout=15)

            if resp.status_code == 200:
                result = resp.json()
                order_id = result.get("orderId", result.get("data", {}).get("orderId"))
                if order_id:
                    log_entry["status"] = "PLACED"
                    log_entry["order_id"] = order_id
                    log_entry["response"] = result
                    self.orders[order_id] = log_entry
                    self.today_orders.append(log_entry)
                    self._log_order(log_entry)
                    logger.info(f"  ✅ Order placed: {order_id}")
                    return str(order_id)
                else:
                    log_entry["status"] = "NO_ORDER_ID"
                    log_entry["response"] = result
                    self._log_order(log_entry)
                    logger.error(f"  ❌ Order placed but no ID returned: {result}")
                    return None
            else:
                log_entry["status"] = f"ERROR_{resp.status_code}"
                log_entry["response"] = resp.text[:500]
                self._log_order(log_entry)
                logger.error(f"  ❌ Order failed [{resp.status_code}]: {resp.text[:300]}")
                return None

        except Exception as e:
            log_entry["status"] = f"EXCEPTION: {e}"
            self._log_order(log_entry)
            logger.error(f"  ❌ Order exception: {e}")
            return None

    # ── Order Modification ───────────────────────────────────
    def modify_order(
        self,
        order_id: str,
        order_type: str = None,
        quantity: int = None,
        price: float = None,
        trigger_price: float = None,
    ) -> bool:
        """Modify an existing order."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": "MODIFY_ORDER",
            "live": self.live,
            "order_id": order_id,
            "order_type": order_type,
            "quantity": quantity,
            "price": price,
            "trigger_price": trigger_price,
        }

        if not self.live:
            log_entry["status"] = "PAPER_MODE"
            self._log_order(log_entry)
            logger.info(f"  [PAPER] Modify order {order_id}")
            return True

        url = f"{API_BASE}/orders/{order_id}"
        payload = {}
        if order_type:
            payload["orderType"] = order_type
        if quantity:
            payload["quantity"] = quantity
        if price is not None:
            payload["price"] = price
        if trigger_price is not None:
            payload["triggerPrice"] = trigger_price
        payload["validity"] = "DAY"
        payload["disclosedQuantity"] = 0

        try:
            resp = requests.put(url, json=payload, headers=_headers(), timeout=15)
            if resp.status_code == 200:
                log_entry["status"] = "MODIFIED"
                self._log_order(log_entry)
                logger.info(f"  ✅ Order modified: {order_id}")
                return True
            else:
                log_entry["status"] = f"ERROR_{resp.status_code}"
                log_entry["response"] = resp.text[:300]
                self._log_order(log_entry)
                logger.error(f"  ❌ Modify failed [{resp.status_code}]: {resp.text[:200]}")
                return False
        except Exception as e:
            log_entry["status"] = f"EXCEPTION: {e}"
            self._log_order(log_entry)
            logger.error(f"  ❌ Modify exception: {e}")
            return False

    # ── Order Cancellation ───────────────────────────────────
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": "CANCEL_ORDER",
            "live": self.live,
            "order_id": order_id,
        }

        if not self.live:
            log_entry["status"] = "PAPER_MODE"
            self._log_order(log_entry)
            logger.info(f"  [PAPER] Cancel order {order_id}")
            return True

        url = f"{API_BASE}/orders/{order_id}"
        try:
            resp = requests.delete(url, headers=_headers(), timeout=15)
            if resp.status_code == 200:
                log_entry["status"] = "CANCELLED"
                self._log_order(log_entry)
                logger.info(f"  ✅ Order cancelled: {order_id}")
                return True
            else:
                log_entry["status"] = f"ERROR_{resp.status_code}"
                self._log_order(log_entry)
                logger.error(f"  ❌ Cancel failed [{resp.status_code}]: {resp.text[:200]}")
                return False
        except Exception as e:
            log_entry["status"] = f"EXCEPTION: {e}"
            self._log_order(log_entry)
            logger.error(f"  ❌ Cancel exception: {e}")
            return False

    # ── Order Status ─────────────────────────────────────────
    def get_order_status(self, order_id: str) -> Optional[dict]:
        """Get status of a specific order."""
        if not self.live:
            return {"orderId": order_id, "orderStatus": "PAPER_TRADED"}

        url = f"{API_BASE}/orders/{order_id}"
        try:
            resp = requests.get(url, headers=_headers(), timeout=15)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.error(f"Order status error: {e}")
        return None

    def get_all_orders(self) -> List[dict]:
        """Get all orders for today."""
        if not self.live:
            return self.today_orders

        url = f"{API_BASE}/orders"
        try:
            resp = requests.get(url, headers=_headers(), timeout=15)
            if resp.status_code == 200:
                return resp.json().get("data", [])
        except Exception as e:
            logger.error(f"Get orders error: {e}")
        return []

    # ── Positions ────────────────────────────────────────────
    def get_positions(self) -> List[dict]:
        """Get current open positions."""
        if not self.live:
            return []

        url = f"{API_BASE}/positions"
        try:
            resp = requests.get(url, headers=_headers(), timeout=15)
            if resp.status_code == 200:
                return resp.json().get("data", [])
        except Exception as e:
            logger.error(f"Positions error: {e}")
        return []

    # ── Portfolio / Holdings ─────────────────────────────────
    def get_holdings(self) -> List[dict]:
        """Get holdings."""
        url = f"{API_BASE}/holdings"
        try:
            resp = requests.get(url, headers=_headers(), timeout=15)
            if resp.status_code == 200:
                return resp.json().get("data", [])
        except Exception as e:
            logger.error(f"Holdings error: {e}")
        return []

    # ── Fund / Margin ────────────────────────────────────────
    def get_fund_limits(self) -> Optional[dict]:
        """Get available margin and fund limits."""
        url = f"{API_BASE}/fundlimit"
        try:
            resp = requests.get(url, headers=_headers(), timeout=15)
            if resp.status_code == 200:
                return resp.json().get("data", resp.json())
        except Exception as e:
            logger.error(f"Fund limits error: {e}")
        return None

    # ── Convenience: Close All Positions ─────────────────────
    def close_all_positions(self, product_type: str = PRODUCT_INTRADAY) -> int:
        """
        Emergency kill switch: Close all open positions by placing opposite orders.
        Returns number of close orders placed.
        """
        positions = self.get_positions()
        closed = 0
        for pos in positions:
            net_qty = int(pos.get("netQty", 0))
            if net_qty == 0:
                continue

            txn = SELL if net_qty > 0 else BUY
            qty = abs(net_qty)
            security_id = pos.get("securityId")

            if not self.live:
                logger.info(f"  [PAPER] Close position: {txn} {qty} secId={security_id}")
                closed += 1
                continue

            order_detail = {
                "transactionType": txn,
                "exchangeSegment": pos.get("exchangeSegment", EXCHANGE_NSE_FO),
                "productType": product_type,
                "orderType": ORDER_MARKET,
                "securityId": str(security_id),
                "quantity": qty,
                "price": 0,
                "triggerPrice": 0,
                "disclosedQuantity": 0,
                "validity": "DAY",
                "afterMarketOrder": False,
                "correlationId": f"close_{datetime.now().strftime('%H%M%S')}",
            }

            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "action": "CLOSE_POSITION",
                "live": self.live,
                "transaction": txn,
                "quantity": qty,
                "security_id": security_id,
            }

            try:
                resp = requests.post(f"{API_BASE}/orders", json=order_detail,
                                     headers=_headers(), timeout=15)
                if resp.status_code == 200:
                    closed += 1
                    log_entry["status"] = "CLOSED"
                    self._log_order(log_entry)
                    logger.info(f"  ✅ Position closed: {txn} {qty}")
                else:
                    log_entry["status"] = f"ERROR_{resp.status_code}"
                    self._log_order(log_entry)
                    logger.error(f"  ❌ Close failed: {resp.text[:200]}")
            except Exception as e:
                log_entry["status"] = f"EXCEPTION: {e}"
                self._log_order(log_entry)
                logger.error(f"  ❌ Close exception: {e}")

        return closed

    # ── Order Logging ────────────────────────────────────────
    def _log_order(self, entry: dict):
        """Append order to daily JSON log file."""
        try:
            existing = []
            if self._order_log_file.exists():
                with open(self._order_log_file, "r") as f:
                    existing = json.load(f)
            existing.append(entry)
            with open(self._order_log_file, "w") as f:
                json.dump(existing, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Order log write failed: {e}")

    # ── Summary ──────────────────────────────────────────────
    def print_summary(self):
        """Print today's order summary."""
        orders = self.get_all_orders()
        if not orders:
            print("  No orders today.")
            return

        print(f"\n  Today's Orders: {len(orders)}")
        for o in orders:
            status = o.get("orderStatus", o.get("status", "?"))
            txn = o.get("transactionType", o.get("transaction", "?"))
            qty = o.get("quantity", 0)
            price = o.get("price", 0)
            print(f"    {txn} {qty} @ {price} — {status}")


# ── CLI Test ──
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    mgr = DhanOrderManager(live=False)

    print("\n=== DHAN ORDER MANAGER TEST ===")
    print(f"  Live Trading Enabled (env): {LIVE_TRADING_ENABLED}")
    print(f"  Manager Mode: {'LIVE' if mgr.live else 'PAPER'}")

    # Test fund limits
    funds = mgr.get_fund_limits()
    if funds:
        print(f"\n  Fund Limits:")
        for k, v in funds.items():
            if isinstance(v, (int, float)):
                print(f"    {k}: ₹{v:,.2f}")

    # Test paper order
    oid = mgr.place_order(
        transaction_type=SELL,
        symbol="nifty",
        quantity=25,
        strike=24500,
        option_type="CE",
        tag="test_order",
    )
    print(f"\n  Test order ID: {oid}")

    # Test positions
    positions = mgr.get_positions()
    print(f"  Open positions: {len(positions)}")

    mgr.print_summary()
    print("\n=== TEST COMPLETE ===")
