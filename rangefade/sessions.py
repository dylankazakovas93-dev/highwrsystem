"""Session labeling (Asia / London / NY sub-sessions) with overnight wrap handling."""
from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

from .config import SessionsCfg


def parse_hhmm(s: str) -> time:
    hh, mm = s.split(":")
    return time(int(hh), int(mm))


def assign_sessions(df: pd.DataFrame, cfg: SessionsCfg) -> pd.DataFrame:
    """Add `session` (str or None) and `tradable` columns. First matching definition wins."""
    df = df.copy()
    idx = df.index
    minutes = idx.hour * 60 + idx.minute
    session = np.full(len(df), None, dtype=object)
    tradable = np.zeros(len(df), dtype=bool)
    unassigned = np.ones(len(df), dtype=bool)
    for d in cfg.definitions:
        s = parse_hhmm(d.start)
        e = parse_hhmm(d.end)
        sm, em = s.hour * 60 + s.minute, e.hour * 60 + e.minute
        if sm < em:
            mask = (minutes >= sm) & (minutes < em)
        else:  # wraps midnight (e.g. asia 18:00 -> 02:00)
            mask = (minutes >= sm) | (minutes < em)
        mask &= unassigned
        session[mask] = d.name
        tradable[mask] = d.tradable and (d.name in cfg.trade)
        unassigned &= ~mask
    df["session"] = session
    df["tradable"] = tradable
    return df


def day_reference_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Per-bar reference prices: prior trading-day close, current day open, RTH (09:30) open."""
    df = df.copy()
    g = df.groupby("trading_day", sort=True)
    day_close = g["close"].last()
    prior_close = day_close.shift(1)
    df["prior_close"] = pd.Series(df["trading_day"].map(prior_close), index=df.index)
    df["day_open"] = g["open"].transform("first")

    minutes = df.index.hour * 60 + df.index.minute
    is_rth = minutes >= (9 * 60 + 30)
    rth_first = (
        df[is_rth & (minutes < 16 * 60)].groupby("trading_day")["open"].first()
        if is_rth.any()
        else pd.Series(dtype=float)
    )
    rth_open = pd.Series(df["trading_day"].map(rth_first), index=df.index)
    rth_open[~is_rth | (minutes >= 16 * 60)] = np.nan  # only defined once RTH has opened
    df["rth_open"] = rth_open
    return df
