"""
Data utilities for fetching and preparing Nifty data.
Supports sample data generation and CSV loading.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def generate_sample_nifty_data(days: int = 90, timeframe: str = "15min") -> pd.DataFrame:
    """
    Generate realistic synthetic Nifty 15-min OHLCV data for backtesting.
    Simulates trending + sideways phases with realistic volume patterns.
    """
    np.random.seed(42)

    intervals_map = {
        "1min": 375,   # 6.25 hrs / 1min
        "5min": 75,    # 6.25 hrs / 5min
        "15min": 25,   # 6.25 hrs / 15min
        "1H": 7,
        "4H": 2,
        "Daily": 1,
    }
    bars_per_day = intervals_map.get(timeframe, 25)
    total_bars = days * bars_per_day

    # Start price around Nifty level
    base_price = 22000.0
    prices = [base_price]

    # Simulate with regime changes (trending / sideways)
    regime_length = bars_per_day * 5  # ~5 days per regime
    trend = 0.0

    for i in range(1, total_bars):
        if i % regime_length == 0:
            # Switch regime
            trend = np.random.choice([-1, 0, 1]) * np.random.uniform(0.5, 2.0)

        noise = np.random.normal(0, 8)  # Intraday noise
        drift = trend * 0.3
        new_price = prices[-1] + drift + noise
        new_price = max(new_price, 18000)  # Floor
        prices.append(new_price)

    # Generate OHLCV
    freq_map = {"5min": "5T", "15min": "15T", "1H": "1H", "4H": "4H", "Daily": "1D"}
    freq = freq_map.get(timeframe, "15T")

    # Build trading timestamps (9:15 to 15:30 IST)
    # Start from (today - days) so the data ends near today
    dates = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 10)  # extra buffer for weekends
    current_date = start_date.replace(hour=9, minute=15, second=0, microsecond=0)
    bar_count = 0

    while bar_count < total_bars:
        if current_date.weekday() < 5:  # Mon-Fri
            trading_start = current_date.replace(hour=9, minute=15)
            for j in range(bars_per_day):
                if bar_count >= total_bars:
                    break
                minutes_offset = j * {"1min": 1, "5min": 5, "15min": 15, "1H": 60, "4H": 240, "Daily": 0}[timeframe]
                bar_time = trading_start + timedelta(minutes=minutes_offset)
                dates.append(bar_time)
                bar_count += 1
        current_date += timedelta(days=1)

    dates = dates[:total_bars]
    prices = prices[:total_bars]

    data = []
    for i, (dt, close) in enumerate(zip(dates, prices)):
        high = close + abs(np.random.normal(0, 15))
        low = close - abs(np.random.normal(0, 15))
        open_price = close + np.random.normal(0, 5)

        # Ensure OHLC consistency
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        # Volume: higher at open/close, random otherwise
        hour = dt.hour
        base_vol = np.random.randint(50000, 200000)
        if hour == 9 or hour == 15:
            base_vol *= 2
        if hour == 12 or hour == 13:
            base_vol = int(base_vol * 0.6)

        data.append({
            "datetime": dt,
            "open": round(open_price, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close, 2),
            "volume": base_vol,
        })

    df = pd.DataFrame(data)
    df.set_index("datetime", inplace=True)
    return df


def load_csv_data(filepath: str) -> pd.DataFrame:
    """
    Load OHLCV data from CSV.
    Expected columns: datetime, open, high, low, close, volume
    """
    df = pd.read_csv(filepath, parse_dates=["datetime"], index_col="datetime")
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df.sort_index(inplace=True)
    return df


def compute_indicators(df: pd.DataFrame, fast_period: int = 9, slow_period: int = 21) -> pd.DataFrame:
    """
    Compute all technical indicators needed for the strategy.
    """
    df = df.copy()

    # --- EMAs ---
    df["ema_fast"] = df["close"].ewm(span=fast_period, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=slow_period, adjust=False).mean()

    # --- EMA Crossover Detection ---
    df["fast_above_slow"] = df["ema_fast"] > df["ema_slow"]
    df["crossover_bull"] = (df["fast_above_slow"]) & (~df["fast_above_slow"].shift(1).fillna(False))
    df["crossover_bear"] = (~df["fast_above_slow"]) & (df["fast_above_slow"].shift(1).fillna(True))

    # --- Volume Filter ---
    df["vol_sma"] = df["volume"].rolling(window=20).mean()
    from config import VOLUME_MULTIPLIER
    if VOLUME_MULTIPLIER > 0:
        df["high_volume"] = df["volume"] > (df["vol_sma"] * VOLUME_MULTIPLIER)
    else:
        df["high_volume"] = True  # Volume filter disabled

    # --- ADX (simplified using ATR-based proxy) ---
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1))
        )
    )
    df["atr"] = df["tr"].rolling(window=14).mean()

    # Simplified ADX proxy: ratio of directional movement to ATR
    df["plus_dm"] = np.where(
        (df["high"] - df["high"].shift(1)) > (df["low"].shift(1) - df["low"]),
        np.maximum(df["high"] - df["high"].shift(1), 0), 0
    )
    df["minus_dm"] = np.where(
        (df["low"].shift(1) - df["low"]) > (df["high"] - df["high"].shift(1)),
        np.maximum(df["low"].shift(1) - df["low"], 0), 0
    )

    df["plus_di"] = 100 * (pd.Series(df["plus_dm"]).ewm(span=14).mean() / df["atr"])
    df["minus_di"] = 100 * (pd.Series(df["minus_dm"]).ewm(span=14).mean() / df["atr"])

    dx = abs(df["plus_di"] - df["minus_di"]) / (df["plus_di"] + df["minus_di"]) * 100
    df["adx"] = dx.ewm(span=14).mean()
    from config import SIDEWAYS_ADX_THRESHOLD
    df["not_sideways"] = df["adx"] > SIDEWAYS_ADX_THRESHOLD  # Trending market

    # --- RSI ---
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # --- Swing High/Low (for SL placement) ---
    df["swing_low"] = df["low"].rolling(window=10, center=True).min()
    df["swing_high"] = df["high"].rolling(window=10, center=True).max()

    # Fill forward swing levels for real-time use
    df["recent_swing_low"] = df["low"].rolling(window=20).min()
    df["recent_swing_high"] = df["high"].rolling(window=20).max()

    return df


def get_atm_strike(spot_price: float, step: int = 50) -> float:
    """Get ATM strike price rounded to nearest step."""
    return round(spot_price / step) * step


def estimate_option_premium(spot: float, strike: float, dte: int,
                             option_type: str = "CE", iv: float = 15.0) -> float:
    """
    Simplified option premium estimation for backtesting.
    Uses a basic intrinsic + time value model (not Black-Scholes for simplicity).
    """
    intrinsic = max(0, spot - strike) if option_type == "CE" else max(0, strike - spot)

    # Time value approximation
    time_value = (iv / 100) * spot * np.sqrt(dte / 365) * 0.4

    premium = intrinsic + time_value
    return round(max(premium, 5), 2)  # Min premium floor
