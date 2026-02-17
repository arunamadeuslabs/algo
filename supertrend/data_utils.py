"""
Data Utilities for Supertrend + VWAP Scalping Strategy
=======================================================
Handles Supertrend computation, VWAP, EMA, and data generation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


# ─── Supertrend Indicator ────────────────────────────────────────────────────

def compute_supertrend(df: pd.DataFrame, period: int = 10,
                       multiplier: float = 3.0) -> pd.DataFrame:
    """
    Compute Supertrend indicator.
    Returns DataFrame with 'supertrend', 'supertrend_direction' columns.
    Direction: 1 = Green (bullish), -1 = Red (bearish)
    """
    df = df.copy()

    # ATR
    hl = df["high"] - df["low"]
    hc = abs(df["high"] - df["close"].shift(1))
    lc = abs(df["low"] - df["close"].shift(1))
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    # Basic upper and lower bands
    hl2 = (df["high"] + df["low"]) / 2
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    # Final bands (with clamping logic)
    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    for i in range(1, len(df)):
        # Skip if current basic bands are NaN (ATR not ready)
        if pd.isna(basic_upper.iloc[i]):
            continue

        # Initialize from basic bands if previous was NaN
        if pd.isna(final_upper.iloc[i - 1]):
            final_upper.iloc[i] = basic_upper.iloc[i]
            final_lower.iloc[i] = basic_lower.iloc[i]
            continue

        # Final Upper Band
        if basic_upper.iloc[i] < final_upper.iloc[i - 1] or \
           df["close"].iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        # Final Lower Band
        if basic_lower.iloc[i] > final_lower.iloc[i - 1] or \
           df["close"].iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

    # Supertrend direction
    for i in range(period, len(df)):
        if i == period:
            if df["close"].iloc[i] <= final_upper.iloc[i]:
                supertrend.iloc[i] = final_upper.iloc[i]
                direction.iloc[i] = -1  # Red / bearish
            else:
                supertrend.iloc[i] = final_lower.iloc[i]
                direction.iloc[i] = 1  # Green / bullish
        else:
            prev_st = supertrend.iloc[i - 1]
            prev_dir = direction.iloc[i - 1]

            if prev_dir == 1:  # Was bullish
                if df["close"].iloc[i] < final_lower.iloc[i]:
                    supertrend.iloc[i] = final_upper.iloc[i]
                    direction.iloc[i] = -1
                else:
                    supertrend.iloc[i] = final_lower.iloc[i]
                    direction.iloc[i] = 1
            else:  # Was bearish
                if df["close"].iloc[i] > final_upper.iloc[i]:
                    supertrend.iloc[i] = final_lower.iloc[i]
                    direction.iloc[i] = 1
                else:
                    supertrend.iloc[i] = final_upper.iloc[i]
                    direction.iloc[i] = -1

    df["supertrend"] = supertrend
    df["supertrend_direction"] = direction
    df["supertrend_green"] = direction == 1
    df["supertrend_red"] = direction == -1
    df["atr"] = atr

    return df


# ─── VWAP ────────────────────────────────────────────────────────────────────

def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Compute VWAP (Volume Weighted Average Price).
    Resets daily (intraday VWAP).
    """
    df = df.copy()
    df["_date"] = df.index.date
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["tp_vol"] = df["typical_price"] * df["volume"]

    vwap = pd.Series(index=df.index, dtype=float)

    for day, group in df.groupby("_date"):
        cum_tp_vol = group["tp_vol"].cumsum()
        cum_vol = group["volume"].cumsum()
        day_vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
        vwap.loc[group.index] = day_vwap

    return vwap


# ─── EMA ─────────────────────────────────────────────────────────────────────

def compute_ema(series: pd.Series, period: int = 9) -> pd.Series:
    """Compute Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


# ─── ATR (standalone) ────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.rolling(window=period).mean()


# ─── Strong Candle Check ────────────────────────────────────────────────────

def is_strong_candle(row: pd.Series, min_body_pct: float = 0.35) -> bool:
    """Check if candle has a strong body (body > min_body_pct of range)."""
    candle_range = row["high"] - row["low"]
    if candle_range <= 0:
        return False
    body = abs(row["close"] - row["open"])
    return (body / candle_range) >= min_body_pct


# ─── Full Indicator Computation ──────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all indicators for the Supertrend + VWAP Scalping strategy.
    Adds: supertrend, vwap, ema_9, and derived signals.
    """
    from config import (
        SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER,
        EMA_PERIOD, MIN_CANDLE_BODY_PCT, MIN_VOLUME_RATIO,
    )

    df = df.copy()

    # --- Supertrend ---
    df = compute_supertrend(df, SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER)

    # --- VWAP ---
    df["vwap"] = compute_vwap(df)

    # --- 9 EMA ---
    df["ema_9"] = compute_ema(df["close"], EMA_PERIOD)

    # --- Volume ---
    df["vol_sma"] = df["volume"].rolling(window=20).mean()
    df["volume_ok"] = df["volume"] >= (df["vol_sma"] * MIN_VOLUME_RATIO)

    # --- Candle Analysis ---
    df["strong_candle"] = df.apply(
        lambda r: is_strong_candle(r, MIN_CANDLE_BODY_PCT), axis=1
    )
    df["bullish_candle"] = (df["close"] > df["open"]) & df["strong_candle"]
    df["bearish_candle"] = (df["close"] < df["open"]) & df["strong_candle"]

    # --- Derived Signals ---
    # Price vs VWAP
    df["above_vwap"] = df["close"] > df["vwap"]
    df["below_vwap"] = df["close"] < df["vwap"]

    # Candle vs 9 EMA
    df["candle_above_ema"] = df["close"] > df["ema_9"]   # Close above EMA
    df["candle_below_ema"] = df["close"] < df["ema_9"]   # Close below EMA

    # Buy signal: Price > VWAP + Supertrend Green + Candle above 9 EMA
    df["buy_signal"] = (
        df["above_vwap"] &
        df["supertrend_green"] &
        df["candle_above_ema"] &
        df["volume_ok"]
    )

    # Sell signal: Price < VWAP + Supertrend Red + Candle below 9 EMA
    df["sell_signal"] = (
        df["below_vwap"] &
        df["supertrend_red"] &
        df["candle_below_ema"] &
        df["volume_ok"]
    )

    return df


# ─── Nifty Data Generation ──────────────────────────────────────────────────

def generate_nifty_data(days: int = 365, timeframe: str = "5min",
                        seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic Nifty 5-min OHLCV data for backtesting.
    Includes trending days, mean-reversion days, and gap opens.
    """
    np.random.seed(seed)

    intervals_map = {"1min": 375, "5min": 75, "15min": 25}
    bars_per_day = intervals_map.get(timeframe, 75)

    # Build trading days
    end_date = datetime.now().replace(hour=9, minute=15, second=0, microsecond=0)
    start_date = end_date - timedelta(days=int(days * 1.5))

    trading_days = []
    current = start_date
    while len(trading_days) < days:
        if current.weekday() < 5:
            trading_days.append(current)
        current += timedelta(days=1)

    # Generate daily parameters
    base_price = 22000.0
    daily_closes = [base_price]
    regime_days = 10
    trend = 0

    for d in range(1, days):
        if d % regime_days == 0:
            trend = np.random.choice([-1, -0.5, 0, 0, 0.5, 1]) * np.random.uniform(0.8, 2.5)

        daily_return = np.random.normal(0, 0.008) + trend * 0.001
        gap = np.random.normal(0, 0.003) if np.random.random() > 0.7 else 0
        new_close = daily_closes[-1] * (1 + daily_return + gap)
        daily_closes.append(new_close)

    # Generate intraday bars
    all_data = []
    minute_step = {"1min": 1, "5min": 5, "15min": 15}.get(timeframe, 5)

    for day_idx, day_date in enumerate(trading_days):
        day_close = daily_closes[day_idx]
        daily_vol = np.random.uniform(0.008, 0.015)

        # Intraday trend bias
        intraday_bias = np.random.normal(0, 0.3)
        price = day_close * (1 + np.random.normal(0, 0.003))

        for bar in range(bars_per_day):
            bar_time = day_date.replace(hour=9, minute=15) + timedelta(minutes=bar * minute_step)

            # Time-of-day volatility pattern
            hour = bar_time.hour
            vol_multiplier = 1.0
            if hour == 9:
                vol_multiplier = 1.8
            elif hour in [14, 15]:
                vol_multiplier = 1.4
            elif hour in [12, 13]:
                vol_multiplier = 0.6

            bar_vol = daily_vol * vol_multiplier / np.sqrt(bars_per_day)
            bar_drift = intraday_bias * 0.0001

            # Generate OHLC
            returns = np.random.normal(bar_drift, bar_vol)
            close = price * (1 + returns)

            intra_high = max(price, close) * (1 + abs(np.random.normal(0, bar_vol * 0.3)))
            intra_low = min(price, close) * (1 - abs(np.random.normal(0, bar_vol * 0.3)))
            open_price = price + np.random.normal(0, 2)

            high = max(intra_high, open_price, close)
            low = min(intra_low, open_price, close)

            # Volume
            base_vol = np.random.randint(80000, 250000)
            if hour == 9 or hour == 15:
                base_vol = int(base_vol * 2.0)
            elif hour in [12, 13]:
                base_vol = int(base_vol * 0.5)

            all_data.append({
                "datetime": bar_time,
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": base_vol,
            })

            price = close

    df = pd.DataFrame(all_data)
    df.set_index("datetime", inplace=True)
    return df
