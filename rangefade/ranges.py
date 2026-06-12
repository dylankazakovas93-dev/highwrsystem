"""Consolidation (range box) detection.

Three methods produce the same per-bar view:
  rolling  — box = high/low of the trailing W-minute window ENDING AT THE PRIOR BAR
             ("established box": the current bar interacts with it but doesn't define it),
             valid when width within bounds and window coverage is sufficient.
  adaptive — rolling + structural requirements baked into validity: window extremes
             must be old (no fresh W-window high/low in the last N minutes) and net
             drift across the window must be small relative to width.
  session  — box frozen from a formation sub-session (e.g. the Asia range), traded
             during later sessions of the same trading day.

A contiguous run of valid bars (small gaps tolerated) forms an *episode*; the
strategy allows one attempt per side per episode.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config

BIG_AGE = 10**6


@dataclass
class RangeView:
    """Per-bar arrays aligned with the source frame (NaN / -1 where undefined)."""

    box_high: np.ndarray
    box_low: np.ndarray
    valid: np.ndarray          # bool
    episode: np.ndarray        # int episode id, -1 when invalid
    win_start: np.ndarray      # int position of first bar inside the box window
    win_end: np.ndarray        # int position one-past-last bar of the box window (exclusive)

    @property
    def box_mid(self) -> np.ndarray:
        return (self.box_high + self.box_low) / 2.0

    @property
    def box_width(self) -> np.ndarray:
        return self.box_high - self.box_low


def _episodes_from_valid(valid: np.ndarray, day_codes: np.ndarray, gap_tolerance: int) -> np.ndarray:
    """Number contiguous valid runs; gaps <= tolerance within the same day keep the episode alive."""
    episode = np.full(len(valid), -1, dtype=np.int64)
    pos = np.nonzero(valid)[0]
    if len(pos) == 0:
        return episode
    eid = 0
    last = pos[0]
    episode[pos[0]] = eid
    for p in pos[1:]:
        if day_codes[p] != day_codes[last] or (p - last) > gap_tolerance + 1:
            eid += 1
        episode[p] = eid
        last = p
    return episode


def detect_ranges(df: pd.DataFrame, cfg: Config) -> RangeView:
    r = cfg.range
    n = len(df)
    bar_min = cfg.data.bar_minutes
    day_codes = pd.factorize(df["trading_day"])[0]

    if r.method in ("rolling", "adaptive"):
        win = f"{r.window_minutes}min"
        g = df.groupby("trading_day", sort=True)
        roll_high = g["high"].rolling(win).max().droplevel(0)
        roll_low = g["low"].rolling(win).min().droplevel(0)
        roll_cnt = g["close"].rolling(win).count().droplevel(0)
        # establish the box on the prior bar
        td = df["trading_day"]
        box_high = roll_high.groupby(td).shift(1).to_numpy()
        box_low = roll_low.groupby(td).shift(1).to_numpy()
        cnt = roll_cnt.groupby(td).shift(1).to_numpy()

        width = box_high - box_low
        need_bars = r.min_window_coverage * (r.window_minutes / bar_min)
        valid = (
            np.isfinite(width)
            & (cnt >= need_bars)
            & (width >= r.min_width_points)
            & (width <= r.max_width_points)
        )

        # window start/end positions (established window ends at the prior bar)
        win_start = np.full(n, -1, dtype=np.int64)
        win_end = np.full(n, -1, dtype=np.int64)
        ts_ns = df.index.asi8
        wdelta = np.int64(r.window_minutes) * 60_000_000_000
        for day in np.unique(day_codes):
            dpos = np.nonzero(day_codes == day)[0]
            dts = ts_ns[dpos]
            # window for bar i covers (ts[i-1] - W, ts[i-1]]
            starts = np.searchsorted(dts, dts - wdelta, side="right")
            win_start[dpos[1:]] = dpos[starts[:-1]]
            win_end[dpos[1:]] = dpos[:-1] + 1

        if r.method == "adaptive":
            high = df["high"].to_numpy()
            low = df["low"].to_numpy()
            close = df["close"].to_numpy()
            new_hi = np.where(np.isfinite(box_high), high > box_high, False)
            new_lo = np.where(np.isfinite(box_low), low < box_low, False)
            age_hi = np.full(n, BIG_AGE, dtype=np.int64)
            age_lo = np.full(n, BIG_AGE, dtype=np.int64)
            idx_all = np.arange(n)
            for day in np.unique(day_codes):
                dpos = np.nonzero(day_codes == day)[0]
                for ev, age in ((new_hi, age_hi), (new_lo, age_lo)):
                    marks = np.where(ev[dpos], idx_all[dpos], -1)
                    last = np.maximum.accumulate(marks)
                    a = np.where(last >= 0, idx_all[dpos] - last, BIG_AGE)
                    age[dpos] = a
            min_age = max(1, r.extremes_min_age_minutes // bar_min)
            ok_age = (age_hi >= min_age) & (age_lo >= min_age)
            drift_ok = np.ones(n, dtype=bool)
            has_win = win_start >= 0
            ws = np.where(has_win, win_start, 0)
            we = np.where(has_win, win_end - 1, 0)
            with np.errstate(invalid="ignore"):
                drift = np.abs(close[we] - close[ws])
                drift_ok = ~has_win | ~np.isfinite(width) | (drift <= r.max_net_drift_frac * width)
            valid &= ok_age & drift_ok

    elif r.method == "session":
        box_high = np.full(n, np.nan)
        box_low = np.full(n, np.nan)
        valid = np.zeros(n, dtype=bool)
        win_start = np.full(n, -1, dtype=np.int64)
        win_end = np.full(n, -1, dtype=np.int64)
        sess = df["session"].to_numpy()
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        for day in np.unique(day_codes):
            dpos = np.nonzero(day_codes == day)[0]
            fpos = dpos[sess[dpos] == r.formation_session]
            if len(fpos) < 30:  # formation session too thin to define a credible box
                continue
            bh = float(np.max(high[fpos]))
            bl = float(np.min(low[fpos]))
            width = bh - bl
            after = dpos[dpos > fpos[-1]]
            box_high[after] = bh
            box_low[after] = bl
            win_start[after] = fpos[0]
            win_end[after] = fpos[-1] + 1
            if r.min_width_points <= width <= r.max_width_points:
                valid[after] = True
    else:
        raise ValueError(f"unknown range.method {r.method}")

    episode = _episodes_from_valid(valid, day_codes, gap_tolerance=5)
    return RangeView(
        box_high=np.asarray(box_high, dtype=float),
        box_low=np.asarray(box_low, dtype=float),
        valid=valid,
        episode=episode,
        win_start=win_start,
        win_end=win_end,
    )


def mid_crossings(close_window: np.ndarray, mid: float) -> int:
    """Count sign flips of (close - mid): proof of two-way rotation through the middle."""
    s = np.sign(close_window - mid)
    s = s[s != 0]
    if len(s) < 2:
        return 0
    return int(np.sum(s[1:] != s[:-1]))
