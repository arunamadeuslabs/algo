"""
Data Utilities for Dual-Confirmation Momentum Strategy
=======================================================
Handles data generation, indicator computation, and market structure analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


# ─── Technical Indicators ────────────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI (Relative Strength Index)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))

    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26,
                 signal: int = 9) -> tuple:
    """
    Compute MACD line, signal line, and histogram.
    Returns (macd_line, signal_line, histogram).
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period).mean()
    return atr


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute simplified ADX."""
    plus_dm = df["high"].diff()
    minus_dm = -df["low"].diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr = compute_atr(df, period)

    plus_di = 100 * plus_dm.ewm(span=period).mean() / atr
    minus_di = 100 * minus_dm.ewm(span=period).mean() / atr

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1)) * 100
    adx = dx.ewm(span=period).mean()
    return adx


def compute_emas(series: pd.Series, fast: int = 9, slow: int = 21) -> tuple:
    """Compute fast and slow EMAs."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    return ema_fast, ema_slow


# ─── Market Structure ────────────────────────────────────────────────────────

def detect_swing_highs(highs: pd.Series, lookback: int = 10) -> pd.Series:
    """Detect swing highs (rolling max)."""
    return highs.rolling(window=lookback, center=False).max()


def detect_swing_lows(lows: pd.Series, lookback: int = 10) -> pd.Series:
    """Detect swing lows (rolling min)."""
    return lows.rolling(window=lookback, center=False).min()


def is_strong_candle(row: pd.Series, min_body_pct: float = 0.4) -> bool:
    """Check if candle has a strong body (body > min_body_pct of range)."""
    candle_range = row["high"] - row["low"]
    if candle_range <= 0:
        return False
    body = abs(row["close"] - row["open"])
    return (body / candle_range) >= min_body_pct


# ─── Full Indicator Computation ──────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame, config=None) -> pd.DataFrame:
    """
    Compute all technical indicators needed for the momentum strategy.
    """
    from config import (
        RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
        EMA_FAST, EMA_SLOW, ATR_PERIOD, VOLUME_SURGE_MULTIPLIER,
        ADX_MIN_TREND, BREAKOUT_LOOKBACK, SWING_LOOKBACK,
        MIN_CANDLE_BODY_PCT,
    )

    df = df.copy()

    # --- EMAs ---
    df["ema_fast"], df["ema_slow"] = compute_emas(df["close"], EMA_FAST, EMA_SLOW)

    # --- RSI ---
    df["rsi"] = compute_rsi(df["close"], RSI_PERIOD)

    # --- MACD ---
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(
        df["close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL
    )

    # MACD crossover detection
    df["macd_bull_cross"] = (df["macd"] > df["macd_signal"]) & \
                             (df["macd"].shift(1) <= df["macd_signal"].shift(1))
    df["macd_bear_cross"] = (df["macd"] < df["macd_signal"]) & \
                             (df["macd"].shift(1) >= df["macd_signal"].shift(1))

    # --- ATR ---
    df["atr"] = compute_atr(df, ATR_PERIOD)

    # --- ADX ---
    df["adx"] = compute_adx(df, 14)
    df["trending"] = df["adx"] > ADX_MIN_TREND

    # --- Volume ---
    df["vol_sma"] = df["volume"].rolling(window=20).mean()
    df["volume_surge"] = df["volume"] > (df["vol_sma"] * VOLUME_SURGE_MULTIPLIER)

    # --- Market Structure ---
    df["swing_high"] = detect_swing_highs(df["high"], SWING_LOOKBACK)
    df["swing_low"] = detect_swing_lows(df["low"], SWING_LOOKBACK)
    df["breakout_high"] = df["high"].rolling(window=BREAKOUT_LOOKBACK).max()
    df["breakout_low"] = df["low"].rolling(window=BREAKOUT_LOOKBACK).min()

    # --- Trend Direction ---
    df["trend_up"] = (df["ema_fast"] > df["ema_slow"]) & (df["close"] > df["ema_slow"])
    df["trend_down"] = (df["ema_fast"] < df["ema_slow"]) & (df["close"] < df["ema_slow"])

    # --- Candle Strength ---
    df["strong_candle"] = df.apply(
        lambda r: is_strong_candle(r, MIN_CANDLE_BODY_PCT), axis=1
    )
    df["bullish_candle"] = (df["close"] > df["open"]) & df["strong_candle"]
    df["bearish_candle"] = (df["close"] < df["open"]) & df["strong_candle"]

    # --- Momentum (% from recent low/high) ---
    recent_low = df["low"].rolling(BREAKOUT_LOOKBACK).min()
    recent_high = df["high"].rolling(BREAKOUT_LOOKBACK).max()
    df["momentum_up"] = (df["close"] - recent_low) / recent_low * 100
    df["momentum_down"] = (recent_high - df["close"]) / recent_high * 100

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
