"""
Data utilities for Sapphire Strategy.
Generates Nifty data and estimates option premiums using Black-Scholes.
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
    iv: annualized implied volatility (decimal, e.g., 0.14 for 14%)
    dte_years: time to expiry in years
    r: risk-free rate (decimal)
    Returns premium in points.
    """
    if dte_years <= 0:
        # At expiry, return intrinsic value
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
    """
    Compute option Greeks using Black-Scholes.
    Returns dict with delta, gamma, theta, vega.
    """
    if dte_years <= 1e-8:
        intrinsic = max(0, spot - strike) if option_type == "CE" else max(0, strike - spot)
        return {"delta": 1.0 if intrinsic > 0 else 0.0, "gamma": 0, "theta": 0, "vega": 0}

    d1 = (np.log(spot / strike) + (r + 0.5 * iv ** 2) * dte_years) / (iv * np.sqrt(dte_years))
    d2 = d1 - iv * np.sqrt(dte_years)

    # Delta
    if option_type == "CE":
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1

    # Gamma
    gamma = norm.pdf(d1) / (spot * iv * np.sqrt(dte_years))

    # Theta (per day)
    term1 = -(spot * norm.pdf(d1) * iv) / (2 * np.sqrt(dte_years))
    if option_type == "CE":
        term2 = -r * strike * np.exp(-r * dte_years) * norm.cdf(d2)
    else:
        term2 = r * strike * np.exp(-r * dte_years) * norm.cdf(-d2)
    theta = (term1 + term2) / 365  # per day

    # Vega (per 1% IV change)
    vega = spot * np.sqrt(dte_years) * norm.pdf(d1) / 100

    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}


# ─── Strike Helpers ─────────────────────────────────────────────────────────

def get_atm_strike(spot: float, step: int = 50) -> float:
    """Round to nearest strike."""
    return round(spot / step) * step


def get_otm_strikes(spot: float, ce_offset: int = 200, pe_offset: int = 200,
                    step: int = 50) -> tuple:
    """Get OTM CE and PE strikes for strangle."""
    atm = get_atm_strike(spot, step)
    ce_strike = atm + ce_offset
    pe_strike = atm - pe_offset
    return ce_strike, pe_strike


# ─── Premium Simulation Engine ──────────────────────────────────────────────

def simulate_intraday_premiums(spot_series: pd.Series, ce_strike: float,
                                pe_strike: float, iv: float = 0.14,
                                r: float = 0.065, dte_days: float = 7.0,
                                entry_time_str: str = "09:20") -> pd.DataFrame:
    """
    Simulate option premiums for CE and PE legs throughout the day
    based on spot movement + time decay.
    
    spot_series: pd.Series with datetime index and spot prices
    Returns DataFrame with columns: ce_premium, pe_premium, ce_delta, pe_delta, theta_ce, theta_pe
    """
    records = []
    
    # Get total minutes in trading day (9:15 to 15:30 = 375 min)
    total_trading_minutes = 375

    for ts, spot in spot_series.items():
        # Calculate intraday time decay
        # Minutes elapsed since 9:15
        minutes_from_open = (ts.hour - 9) * 60 + (ts.minute - 15)
        minutes_from_open = max(0, min(minutes_from_open, total_trading_minutes))
        
        # Fraction of day elapsed
        day_fraction = minutes_from_open / total_trading_minutes
        
        # Remaining DTE (decreases throughout the day)
        remaining_dte = max(dte_days - day_fraction, 0.01) / 365.0

        # Add intraday IV variation (typically higher at open/close)
        iv_adj = iv
        if ts.hour == 9:
            iv_adj = iv * 1.10  # 10% higher at open
        elif ts.hour >= 15:
            iv_adj = iv * 1.05  # 5% higher near close
        elif ts.hour in [12, 13]:
            iv_adj = iv * 0.95  # 5% lower midday

        ce_price = black_scholes_price(spot, ce_strike, remaining_dte, iv_adj, r, "CE")
        pe_price = black_scholes_price(spot, pe_strike, remaining_dte, iv_adj, r, "PE")

        greeks_ce = bs_greeks(spot, ce_strike, remaining_dte, iv_adj, r, "CE")
        greeks_pe = bs_greeks(spot, pe_strike, remaining_dte, iv_adj, r, "PE")

        records.append({
            "datetime": ts,
            "spot": spot,
            "ce_premium": round(ce_price, 2),
            "pe_premium": round(pe_price, 2),
            "ce_delta": round(greeks_ce["delta"], 4),
            "pe_delta": round(greeks_pe["delta"], 4),
            "ce_theta": round(greeks_ce["theta"], 4),
            "pe_theta": round(greeks_pe["theta"], 4),
            "ce_gamma": round(greeks_ce["gamma"], 6),
            "pe_gamma": round(greeks_pe["gamma"], 6),
        })

    df = pd.DataFrame(records)
    df.set_index("datetime", inplace=True)
    return df


# ─── Nifty Data Generation ──────────────────────────────────────────────────

def generate_nifty_data(days: int = 365, timeframe: str = "5min",
                        seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic Nifty 5-min OHLCV data for 1-year backtest.
    Includes regime changes, gap opens, expiry volatility.
    """
    np.random.seed(seed)

    intervals_map = {
        "1min": 375,
        "5min": 75,
        "15min": 25,
    }
    bars_per_day = intervals_map.get(timeframe, 75)
    
    # Build trading days (exclude weekends)
    end_date = datetime.now().replace(hour=9, minute=15, second=0, microsecond=0)
    start_date = end_date - timedelta(days=int(days * 1.5))  # buffer for weekends/holidays
    
    trading_days = []
    current = start_date
    while len(trading_days) < days:
        if current.weekday() < 5:
            trading_days.append(current)
        current += timedelta(days=1)
    
    # Generate daily levels first (for realistic gap opens)
    base_price = 22000.0
    daily_closes = [base_price]
    daily_ivs = [13.0]  # IV starts at 13%
    daily_ranges = [0]  # Intraday range in points
    
    # Regime parameters
    regime_days = 12
    trend = 0
    
    for d in range(1, days):
        if d % regime_days == 0:
            trend = np.random.choice([-1, -0.5, 0, 0, 0.5, 1]) * np.random.uniform(0.5, 2.0)
        
        # Daily return with realistic distribution
        # Nifty daily std ~ 1.0-1.5% of price
        base_std = daily_closes[-1] * 0.011  # ~1.1% daily std
        
        # Occasionally large moves (fat tails)
        if np.random.random() < 0.06:  # ~6% chance of big move day
            daily_return = np.random.choice([-1, 1]) * np.random.uniform(1.5, 3.0) * base_std
        elif np.random.random() < 0.15:  # 15% chance of moderate swing
            daily_return = np.random.choice([-1, 1]) * np.random.uniform(1.0, 1.8) * base_std
        else:
            daily_return = trend * 0.2 + np.random.normal(0, 1.0) * base_std * 0.6
        
        new_close = daily_closes[-1] + daily_return
        new_close = max(new_close, 18000)
        daily_closes.append(new_close)
        
        # Track intraday range
        intraday_range = abs(daily_return) * np.random.uniform(1.2, 2.5)
        daily_ranges.append(intraday_range)
        
        # IV mean-reversion with realistic dynamics
        iv = daily_ivs[-1] + np.random.normal(0, 0.4)
        iv = 13.0 + (iv - 13.0) * 0.95  # Mean-revert to 13%
        
        # Spike IV on big moves (fear premium)
        move_pct = abs(daily_return) / daily_closes[-1] * 100
        if move_pct > 1.5:
            iv += move_pct * 1.5  # Big IV spike on large moves
        elif move_pct > 0.8:
            iv += move_pct * 0.5
        
        # Expiry day (Thursday) tends to have higher IV
        if trading_days[d].weekday() == 3:
            iv += 1.5
        
        iv = max(10, min(30, iv))  # IV between 10-30%
        daily_ivs.append(iv)
    
    # Now generate intraday bars
    all_bars = []
    minute_step = {"1min": 1, "5min": 5, "15min": 15}[timeframe]
    
    for d_idx, day in enumerate(trading_days):
        day_open = daily_closes[d_idx]
        if d_idx > 0:
            # Gap open: based on daily close change
            gap = np.random.normal(0, daily_closes[d_idx] * 0.002)
            day_open = daily_closes[d_idx - 1] + gap
        
        prev_close = day_open
        iv_today = daily_ivs[d_idx]
        
        # Scale intraday noise based on the day's range
        day_range = daily_ranges[d_idx] if d_idx < len(daily_ranges) else 150
        bar_noise_scale = max(4, day_range / (bars_per_day * 0.5))
        
        for bar_idx in range(bars_per_day):
            bar_time = day.replace(hour=9, minute=15) + timedelta(minutes=bar_idx * minute_step)
            
            # Intraday volatility pattern (U-shaped)
            hour = bar_time.hour
            minute = bar_time.minute
            
            if hour == 9 and minute <= 30:
                vol_mult = 2.2  # High vol at open
            elif hour >= 15:
                vol_mult = 1.6  # Higher vol near close
            elif hour in [12, 13]:
                vol_mult = 0.5  # Low vol midday
            else:
                vol_mult = 1.0
            
            # Bar-level price movement (scaled by day's range)
            noise = np.random.normal(0, bar_noise_scale * vol_mult)
            
            # Add trending within the day (market doesn't random walk)
            if bar_idx > 3:
                intraday_trend = (day_open + (daily_closes[d_idx] - day_open) * bar_idx / bars_per_day) - prev_close
                noise += intraday_trend * 0.15
            
            close = prev_close + noise
            
            # Tend towards daily close
            if bar_idx > bars_per_day * 0.7:
                pull = (daily_closes[d_idx] - close) * 0.03
                close += pull
            
            high = close + abs(np.random.normal(0, bar_noise_scale * vol_mult * 1.2))
            low = close - abs(np.random.normal(0, bar_noise_scale * vol_mult * 1.2))
            open_p = prev_close + np.random.normal(0, bar_noise_scale * 0.3)
            
            high = max(high, open_p, close)
            low = min(low, open_p, close)
            
            # Volume
            base_vol = np.random.randint(30000, 150000)
            if hour == 9:
                base_vol = int(base_vol * 2.5)
            elif hour >= 15:
                base_vol = int(base_vol * 1.8)
            elif hour in [12, 13]:
                base_vol = int(base_vol * 0.5)
            
            # Expiry day volume boost
            if day.weekday() == 3:
                base_vol = int(base_vol * 1.5)
            
            all_bars.append({
                "datetime": bar_time,
                "open": round(open_p, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": base_vol,
                "daily_iv": round(iv_today, 2),
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
    
    # Group by date
    df["date"] = df.index.date
    grouped = df.groupby("date")
    
    vwap = grouped["tp_vol"].cumsum() / grouped["volume"].cumsum()
    return vwap


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
