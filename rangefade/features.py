"""Per-bar features: VWAP (+slope), prior-day ATR, realized vol, day move, running extremes.

All filters consume these columns. NaN feature values cause the filter to BLOCK the
trade (recorded as a warmup skip) — we never trade on unverifiable conditions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def _grouped(df: pd.DataFrame):
    return df.groupby("trading_day", sort=True, group_keys=False)


def add_vwap(df: pd.DataFrame, anchor: str) -> pd.DataFrame:
    """anchor: 'day' (18:00 roll), 'rth' (09:30), or 'auto' (rth when available, else day)."""
    df = df.copy()
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = (tp * df["volume"]).fillna(0.0)

    def _vwap(group_mask: pd.Series) -> pd.Series:
        cum_pv = pv.where(group_mask, 0.0).groupby(df["trading_day"]).cumsum()
        cum_v = df["volume"].where(group_mask, 0.0).groupby(df["trading_day"]).cumsum()
        return (cum_pv / cum_v.replace(0, np.nan)).where(group_mask)

    all_mask = pd.Series(True, index=df.index)
    vwap_day = _vwap(all_mask)
    minutes = df.index.hour * 60 + df.index.minute
    rth_mask = pd.Series((minutes >= 9 * 60 + 30) & (minutes < 16 * 60), index=df.index)
    vwap_rth = _vwap(rth_mask)

    if anchor == "day":
        df["vwap"] = vwap_day
    elif anchor == "rth":
        df["vwap"] = vwap_rth
    else:  # auto
        df["vwap"] = vwap_rth.where(rth_mask & vwap_rth.notna(), vwap_day)
    return df


def add_vwap_slope(df: pd.DataFrame, window_minutes: int, bar_minutes: int) -> pd.DataFrame:
    df = df.copy()
    shift_bars = max(1, window_minutes // bar_minutes)
    df["vwap_slope"] = (df["vwap"] - _grouped(df)["vwap"].shift(shift_bars)).abs()
    return df


def add_prior_day_atr(df: pd.DataFrame, atr_days: int) -> pd.DataFrame:
    """ATR over daily bars built on trading_day; mapped back per-bar as PRIOR day's value."""
    df = df.copy()
    g = df.groupby("trading_day", sort=True)
    daily = pd.DataFrame({"high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last()})
    prev_close = daily["close"].shift(1)
    tr = pd.concat(
        [daily["high"] - daily["low"], (daily["high"] - prev_close).abs(), (daily["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(atr_days, min_periods=max(3, atr_days // 3)).mean()
    df["prior_day_atr"] = pd.Series(df["trading_day"].map(atr.shift(1)), index=df.index)
    return df


def add_realized_vol(df: pd.DataFrame, rv_window_minutes: int, median_days: int, bar_minutes: int) -> pd.DataFrame:
    """RV = rolling std (points) of 1-bar close changes; compared to its multi-day median."""
    df = df.copy()
    w = max(5, rv_window_minutes // bar_minutes)
    chg = _grouped(df)["close"].apply(lambda s: s.diff())
    df["rv"] = chg.rolling(w, min_periods=w).std()
    df["rv_median"] = df["rv"].rolling(f"{median_days}D", min_periods=w * 50).median()
    df["rv_ratio"] = df["rv"] / df["rv_median"]
    return df


def add_day_move(df: pd.DataFrame, refs: list[str]) -> pd.DataFrame:
    """Max abs % move vs the configured reference prices (prior_close / rth_open / day_open)."""
    df = df.copy()
    moves = []
    for ref in refs:
        base = df[ref]
        if ref == "rth_open":  # before RTH opens, fall back to the day open
            base = base.fillna(df["day_open"])
        moves.append((df["close"] / base - 1.0).abs() * 100.0)
    df["day_move_pct"] = pd.concat(moves, axis=1).max(axis=1) if moves else 0.0
    return df


def add_running_extremes(df: pd.DataFrame, ext_window_minutes: int, bar_minutes: int) -> pd.DataFrame:
    """Session (trading-day) running high/low and how much they extended over the last M minutes."""
    df = df.copy()
    df["run_max"] = _grouped(df)["high"].cummax()
    df["run_min"] = _grouped(df)["low"].cummin()
    m = max(1, ext_window_minutes // bar_minutes)
    run_max_then = _grouped(df)["run_max"].shift(m)
    run_min_then = _grouped(df)["run_min"].shift(m)
    day_first_max = _grouped(df)["run_max"].transform("first")
    day_first_min = _grouped(df)["run_min"].transform("first")
    df["ext_up"] = df["run_max"] - run_max_then.fillna(day_first_max)
    df["ext_dn"] = run_min_then.fillna(day_first_min) - df["run_min"]
    return df


def load_news_blocks(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Boolean per-bar mask of news-blocked times from an optional calendar CSV."""
    f = cfg.filters
    blocked = np.zeros(len(df), dtype=bool)
    if not f.news_file:
        return blocked
    cal = pd.read_csv(f.news_file)
    cal.columns = [c.strip().lower() for c in cal.columns]
    tcol = next((c for c in ("timestamp", "datetime", "time", "date") if c in cal.columns), None)
    if tcol is None:
        raise ValueError("news file needs a timestamp/datetime column")
    ts = pd.to_datetime(cal[tcol])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(f.news_tz, ambiguous="NaT", nonexistent="NaT")
    ts = ts.dt.tz_convert(cfg.data.tz_market)
    rank = {"low": 0, "medium": 1, "high": 2}
    if "impact" in cal.columns:
        keep = cal["impact"].astype(str).str.lower().map(rank).fillna(2) >= rank.get(f.news_min_impact, 2)
        ts = ts[keep.values]
    idx = df.index
    for t in ts.dropna():
        lo = idx.searchsorted(t - pd.Timedelta(minutes=f.news_block_before_min), side="left")
        hi = idx.searchsorted(t + pd.Timedelta(minutes=f.news_block_after_min), side="right")
        blocked[lo:hi] = True
    return blocked


def compute_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Run the full feature stack. Input must already have trading_day, session, ref prices."""
    f = cfg.filters
    bm = cfg.data.bar_minutes
    df = add_vwap(df, f.vwap_anchor)
    df = add_vwap_slope(df, f.vwap_slope_window_minutes, bm)
    df = add_prior_day_atr(df, f.prior_atr_days)
    df = add_realized_vol(df, f.rv_window_minutes, f.rv_median_days, bm)
    df = add_day_move(df, f.day_move_refs)
    df = add_running_extremes(df, f.extreme_extension_window_minutes, bm)
    return df
