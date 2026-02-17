"""
Data utilities for Iron Condor Strategy.
Generates Bank Nifty data and estimates option premiums using Black-Scholes.
Includes RSI, Bollinger Bands, VIX simulation for entry filters.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
import os


# ─── Black-Scholes Option Pricing ───────────────────────────────────────────

def black_scholes_price(spot: float, strike: float, dte_years: float,
                        iv: float, r: float, option_type: str = "CE") -> float:
    """
    Black-Scholes option price.
    iv: annualized implied volatility (decimal, e.g., 0.16 for 16%)
    dte_years: time to expiry in years
    r: risk-free rate (decimal)
    Returns premium in points.
    """
    if dte_years <= 0:
        if option_type == "CE":
            return max(0, spot - strike)
        else:
            return max(0, strike - spot)

    d1 = (np.log(spot / strike) + (r + 0.5 * iv ** 2) * dte_years) / (iv * np.sqrt(dte_years))
    d2 = d1 - iv * np.sqrt(dte_years)

    if option_type == "CE":
        price = spot * norm.cdf(d1) - strike * np.exp(-r * dte_years) * norm.cdf(d2)
    else:
        price = strike * np.exp(-r * dte_years) * norm.cdf(-d2) - spot * norm.cdf(-d1)

    return max(price, 0.05)


def bs_greeks(spot: float, strike: float, dte_years: float,
              iv: float, r: float, option_type: str = "CE") -> dict:
    """Compute option Greeks using Black-Scholes."""
    if dte_years <= 1e-8:
        intrinsic = max(0, spot - strike) if option_type == "CE" else max(0, strike - spot)
        return {"delta": 1.0 if intrinsic > 0 else 0.0, "gamma": 0, "theta": 0, "vega": 0}

    d1 = (np.log(spot / strike) + (r + 0.5 * iv ** 2) * dte_years) / (iv * np.sqrt(dte_years))
    d2 = d1 - iv * np.sqrt(dte_years)

    if option_type == "CE":
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1

    gamma = norm.pdf(d1) / (spot * iv * np.sqrt(dte_years))

    term1 = -(spot * norm.pdf(d1) * iv) / (2 * np.sqrt(dte_years))
    if option_type == "CE":
        term2 = -r * strike * np.exp(-r * dte_years) * norm.cdf(d2)
    else:
        term2 = r * strike * np.exp(-r * dte_years) * norm.cdf(-d2)
    theta = (term1 + term2) / 365

    vega = spot * np.sqrt(dte_years) * norm.pdf(d1) / 100

    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}


# ─── Strike Helpers ─────────────────────────────────────────────────────────

def get_atm_strike(spot: float, step: int = 100) -> float:
    """Round to nearest Bank Nifty strike."""
    return round(spot / step) * step


def get_iron_condor_strikes(spot: float, ce_offset: int = 500,
                             pe_offset: int = 500, wing_width: int = 500,
                             step: int = 100) -> tuple:
    """
    Get Iron Condor strikes.
    Returns: (sell_ce, buy_ce, sell_pe, buy_pe)
    sell_ce = ATM + ce_offset (short call)
    buy_ce = sell_ce + wing_width (long call — protection)
    sell_pe = ATM - pe_offset (short put)
    buy_pe = sell_pe - wing_width (long put — protection)
    """
    atm = get_atm_strike(spot, step)
    sell_ce = atm + ce_offset
    buy_ce = sell_ce + wing_width
    sell_pe = atm - pe_offset
    buy_pe = sell_pe - wing_width
    return sell_ce, buy_ce, sell_pe, buy_pe


# ─── Technical Indicators ──────────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI (Relative Strength Index)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_bollinger_bands(series: pd.Series, period: int = 20,
                            std_mult: float = 2.0) -> pd.DataFrame:
    """Compute Bollinger Bands. Returns DataFrame with sma, upper, lower, width."""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / sma  # Normalized width
    return pd.DataFrame({
        "bb_sma": sma,
        "bb_upper": upper,
        "bb_lower": lower,
        "bb_width": width,
    }, index=series.index)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


# ─── Premium Simulation Engine ──────────────────────────────────────────────

def simulate_ic_premiums(spot_series: pd.Series, sell_ce: float, buy_ce: float,
                          sell_pe: float, buy_pe: float, iv: float = 0.16,
                          r: float = 0.065, dte_days: float = 3.0) -> pd.DataFrame:
    """
    Simulate option premiums for all 4 Iron Condor legs throughout the day.
    Returns DataFrame with premiums for each leg.
    """
    records = []
    total_trading_minutes = 375  # 9:15 to 15:30

    for ts, spot in spot_series.items():
        minutes_from_open = (ts.hour - 9) * 60 + (ts.minute - 15)
        minutes_from_open = max(0, min(minutes_from_open, total_trading_minutes))
        day_fraction = minutes_from_open / total_trading_minutes
        remaining_dte = max(dte_days - day_fraction, 0.01) / 365.0

        # Intraday IV variation
        iv_adj = iv
        if ts.hour == 9:
            iv_adj = iv * 1.10
        elif ts.hour >= 15:
            iv_adj = iv * 1.05
        elif ts.hour in [12, 13]:
            iv_adj = iv * 0.95

        sell_ce_prem = black_scholes_price(spot, sell_ce, remaining_dte, iv_adj, r, "CE")
        buy_ce_prem = black_scholes_price(spot, buy_ce, remaining_dte, iv_adj, r, "CE")
        sell_pe_prem = black_scholes_price(spot, sell_pe, remaining_dte, iv_adj, r, "PE")
        buy_pe_prem = black_scholes_price(spot, buy_pe, remaining_dte, iv_adj, r, "PE")

        records.append({
            "datetime": ts,
            "spot": spot,
            "sell_ce_prem": round(sell_ce_prem, 2),
            "buy_ce_prem": round(buy_ce_prem, 2),
            "sell_pe_prem": round(sell_pe_prem, 2),
            "buy_pe_prem": round(buy_pe_prem, 2),
            "net_credit_ce": round(sell_ce_prem - buy_ce_prem, 2),
            "net_credit_pe": round(sell_pe_prem - buy_pe_prem, 2),
        })

    df = pd.DataFrame(records)
    df.set_index("datetime", inplace=True)
    return df


# ─── Bank Nifty Data Generation ────────────────────────────────────────────

def generate_banknifty_data(days: int = 365, timeframe: str = "5min",
                             seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic Bank Nifty 5-min OHLCV data for backtest.
    Bank Nifty has higher volatility than Nifty (~1.3-1.8% daily).
    Includes regime changes, gap opens, expiry volatility, VIX simulation.
    """
    np.random.seed(seed)

    intervals_map = {"1min": 375, "5min": 75, "15min": 25}
    bars_per_day = intervals_map.get(timeframe, 75)

    # Build trading days (exclude weekends)
    end_date = datetime.now().replace(hour=9, minute=15, second=0, microsecond=0)
    start_date = end_date - timedelta(days=int(days * 1.5))

    trading_days = []
    current = start_date
    while len(trading_days) < days:
        if current.weekday() < 5:
            trading_days.append(current)
        current += timedelta(days=1)

    # Generate daily levels
    base_price = 50000.0  # Bank Nifty around 50,000
    daily_closes = [base_price]
    daily_ivs = [16.0]  # Bank Nifty IV starts at 16%
    daily_ranges = [0]
    daily_vix = [14.0]  # India VIX starts at 14

    regime_days = 10
    trend = 0

    for d in range(1, days):
        if d % regime_days == 0:
            trend = np.random.choice([-1, -0.5, 0, 0, 0.5, 1]) * np.random.uniform(0.5, 2.5)

        # Bank Nifty daily std ~ 1.3-1.8% (higher than Nifty)
        base_std = daily_closes[-1] * 0.015

        if np.random.random() < 0.07:  # 7% chance of big move
            daily_return = np.random.choice([-1, 1]) * np.random.uniform(1.5, 3.5) * base_std
        elif np.random.random() < 0.15:
            daily_return = np.random.choice([-1, 1]) * np.random.uniform(1.0, 2.0) * base_std
        else:
            daily_return = trend * 0.25 + np.random.normal(0, 1.0) * base_std * 0.6

        new_close = daily_closes[-1] + daily_return
        new_close = max(new_close, 40000)
        daily_closes.append(new_close)

        intraday_range = abs(daily_return) * np.random.uniform(1.2, 2.5)
        daily_ranges.append(intraday_range)

        # IV mean-reversion
        iv = daily_ivs[-1] + np.random.normal(0, 0.5)
        iv = 16.0 + (iv - 16.0) * 0.93
        move_pct = abs(daily_return) / daily_closes[-1] * 100
        if move_pct > 1.5:
            iv += move_pct * 2.0
        elif move_pct > 0.8:
            iv += move_pct * 0.6

        # Expiry day Thursday higher IV
        if trading_days[d].weekday() == 3:
            iv += 2.0

        iv = max(12, min(35, iv))
        daily_ivs.append(iv)

        # VIX simulation (correlated with IV and big moves)
        vix = daily_vix[-1] + np.random.normal(0, 0.3)
        vix = 14.0 + (vix - 14.0) * 0.92  # Mean-revert to 14
        if move_pct > 1.5:
            vix += move_pct * 1.5
        elif move_pct > 0.8:
            vix += move_pct * 0.4
        vix = max(10, min(35, vix))
        daily_vix.append(vix)

    # Generate intraday bars
    all_bars = []
    minute_step = {"1min": 1, "5min": 5, "15min": 15}[timeframe]

    for d_idx, day in enumerate(trading_days):
        day_open = daily_closes[d_idx]
        if d_idx > 0:
            gap = np.random.normal(0, daily_closes[d_idx] * 0.003)
            day_open = daily_closes[d_idx - 1] + gap

        prev_close = day_open
        iv_today = daily_ivs[d_idx]
        vix_today = daily_vix[d_idx]

        day_range = daily_ranges[d_idx] if d_idx < len(daily_ranges) else 300
        bar_noise_scale = max(6, day_range / (bars_per_day * 0.5))

        for bar_idx in range(bars_per_day):
            bar_time = day.replace(hour=9, minute=15) + timedelta(minutes=bar_idx * minute_step)

            hour = bar_time.hour
            minute = bar_time.minute

            if hour == 9 and minute <= 30:
                vol_mult = 2.5  # Higher vol at open (Bank Nifty is volatile)
            elif hour >= 15:
                vol_mult = 1.8
            elif hour in [12, 13]:
                vol_mult = 0.5
            else:
                vol_mult = 1.0

            noise = np.random.normal(0, bar_noise_scale * vol_mult)

            if bar_idx > 3:
                intraday_trend = (day_open + (daily_closes[d_idx] - day_open) * bar_idx / bars_per_day) - prev_close
                noise += intraday_trend * 0.15

            close = prev_close + noise

            if bar_idx > bars_per_day * 0.7:
                pull = (daily_closes[d_idx] - close) * 0.03
                close += pull

            high = close + abs(np.random.normal(0, bar_noise_scale * vol_mult * 1.3))
            low = close - abs(np.random.normal(0, bar_noise_scale * vol_mult * 1.3))
            open_p = prev_close + np.random.normal(0, bar_noise_scale * 0.3)

            high = max(high, open_p, close)
            low = min(low, open_p, close)

            base_vol = np.random.randint(20000, 120000)
            if hour == 9:
                base_vol = int(base_vol * 2.5)
            elif hour >= 15:
                base_vol = int(base_vol * 2.0)
            elif hour in [12, 13]:
                base_vol = int(base_vol * 0.4)

            if day.weekday() == 3:
                base_vol = int(base_vol * 1.8)

            all_bars.append({
                "datetime": bar_time,
                "open": round(open_p, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": base_vol,
                "daily_iv": round(iv_today, 2),
                "vix": round(vix_today, 2),
            })

            prev_close = close

    df = pd.DataFrame(all_bars)
    df.set_index("datetime", inplace=True)
    return df


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Compute VWAP for each trading day."""
    df = df.copy()
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["tp_vol"] = df["typical_price"] * df["volume"]
    df["date"] = df.index.date
    grouped = df.groupby("date")
    vwap = grouped["tp_vol"].cumsum() / grouped["volume"].cumsum()
    return vwap
