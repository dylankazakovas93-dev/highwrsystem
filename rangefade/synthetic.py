"""Synthetic NQ-like 1-min data for ENGINE VALIDATION ONLY.

Regime-switching generator: balance segments are Ornstein-Uhlenbeck (mean-reverting
around a level), trend segments drift. Because balance segments mean-revert BY
CONSTRUCTION, fade strategies will look good on this data — numbers produced from it
say nothing about a real edge. It exists so the whole pipeline can be exercised,
unit-tested and demoed before real data arrives.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic(
    n_days: int = 120,
    seed: int = 42,
    start_price: float = 21500.0,
    start_date: str = "2025-01-06",
    tz: str = "America/New_York",
    substeps: int = 4,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start_date, periods=n_days)
    rows_ts = []
    rows_ohlcv = []
    px = start_price

    for d in days:
        day_start = (d - pd.Timedelta(days=1)).replace(hour=18, minute=0)
        ts = pd.date_range(day_start, periods=23 * 60, freq="min", tz=tz)
        n = len(ts)
        minutes = ts.hour * 60 + ts.minute
        is_rth = (minutes >= 570) & (minutes < 960)
        sigma = np.where(is_rth, 3.2, 1.5) / np.sqrt(substeps)

        # regime plan for the day: rotational vs trend day
        trend_day = rng.random() < 0.30
        seg_len = rng.integers(120, 300)
        regime = np.zeros(n, dtype=int)  # 0 balance, 1 trend
        i = 0
        while i < n:
            ln = int(rng.integers(90, 360))
            if trend_day:
                regime[i:i + ln] = int(rng.random() < 0.5)
            else:
                regime[i:i + ln] = int(rng.random() < 0.15)
            i += ln
        drift_dir = rng.choice([-1.0, 1.0])
        theta = 0.03  # OU pull per substep toward the balance center
        center = px

        opens = np.empty(n); highs = np.empty(n); lows = np.empty(n); closes = np.empty(n)
        for j in range(n):
            if j > 0 and regime[j] != regime[j - 1]:
                center = px  # re-anchor balance around current price
            o_ = px
            hi = px
            lo = px
            for _ in range(substeps):
                shock = rng.normal(0.0, sigma[j])
                if regime[j] == 0:
                    px = px + theta * (center - px) + shock
                else:
                    px = px + drift_dir * 0.15 + shock
                hi = max(hi, px)
                lo = min(lo, px)
            opens[j] = o_; highs[j] = hi; lows[j] = lo; closes[j] = px

        base_vol = np.where(is_rth, 900.0, 150.0)
        vol = (base_vol * rng.lognormal(0.0, 0.6, n)).round().astype(float)
        rows_ts.append(ts)
        rows_ohlcv.append(np.column_stack([opens, highs, lows, closes, vol]))
        px = closes[-1] + rng.normal(0, 8)

    data = np.vstack(rows_ohlcv)
    df = pd.DataFrame(data, columns=["open", "high", "low", "close", "volume"])
    df.index = rows_ts[0].append(rows_ts[1:])  # preserves tz and time unit
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df.index.name = "ts"
    # round to NQ tick
    for col in ("open", "high", "low", "close"):
        df[col] = (df[col] / 0.25).round() * 0.25
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    return df
