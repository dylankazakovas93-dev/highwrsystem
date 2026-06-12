"""Shared test fixtures: deterministic bar construction and a lab config with
all market-state filters disabled (so tests exercise mechanics, not filters)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from rangefade.config import Config, load_config


def make_bars(
    closes,
    start: str = "2025-03-04 09:30",
    tz: str = "America/New_York",
    hi_off: float = 1.0,
    lo_off: float = 1.0,
    volume: float = 500.0,
) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    idx = pd.date_range(start, periods=len(closes), freq="min", tz=tz)
    open_ = np.r_[closes[0], closes[:-1]]
    high = np.maximum(open_, closes) + hi_off
    low = np.minimum(open_, closes) - lo_off
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": closes, "volume": volume},
        index=idx,
    )
    df.index.name = "ts"
    return df


def lab_config() -> Config:
    cfg = load_config()
    cfg.instrument.commission_per_side = 0.0
    cfg.instrument.slippage_ticks = 0
    cfg.range.window_minutes = 30
    cfg.range.min_width_points = 10.0
    cfg.range.max_width_points = 200.0
    f = cfg.filters
    f.max_day_move_pct = None
    f.max_range_vs_prior_atr = None
    f.max_vwap_slope_points = None
    f.rv_max_vs_median = None
    f.extreme_extension_points = None
    f.min_mid_crossings = 0
    cfg.entry.max_trades_per_day = 10
    cfg.entry.stop_after_daily_loss = False
    cfg.exit.tp_points = 15.0
    cfg.exit.min_tp_points = 15.0
    cfg.exit.max_holding_minutes = 600
    cfg.exit.flat_by = None
    return cfg


def wave(low: float, high: float, period: int, n: int, phase: int = 0):
    """Triangle wave of closes between low and high."""
    out = []
    half = period // 2
    amp = high - low
    for i in range(n):
        k = (i + phase) % period
        frac = k / half if k <= half else (period - k) / half
        out.append(low + amp * frac)
    return out
